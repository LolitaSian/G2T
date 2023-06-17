import os
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from model.modeling_gt2 import MyBartForConditionalGeneration as MyBart
from data import WebNLGDataLoader, WebNLGDataset, evaluate_bleu
from tqdm import tqdm, trange
from eval_wqpq.eval import eval

writer = SummaryWriter('./tensorboard_log')


def run(args, logger):
    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_path)
    config = BartConfig.from_pretrained(args.tokenizer_path)
    train_dataset = WebNLGDataset(logger, args, args.train_file, tokenizer)
    dev_dataset = WebNLGDataset(logger, args, args.predict_file, tokenizer)
    train_dataloader = WebNLGDataLoader(args, train_dataset, "train")
    dev_dataloader = WebNLGDataLoader(args, dev_dataset, "dev")
    config.gat_pad_num = args.gat_pad_num
    config.max_input_length = args.max_input_length
    config.max_node_length = args.max_node_length
    
    if args.do_train:
        model = MyBart.from_pretrained(args.model_path, config=config)
        logger.info('model parameters: ' + str(model.num_parameters()))
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if not args.no_lr_decay:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=1000000)
        # 训练
        train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer)

    if args.do_predict:
        # Inference on the test set
        checkpoint = args.output_dir
        model = MyBart.from_pretrained(checkpoint)
        logger.info("Loading checkpoint from {}".format(checkpoint))
        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()
        scores = inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=True)
        logger.info("%s on data: %.4f" % ("BLUE-4", scores['Bleu_4']))


def train(args, logger, model, train_dataloader, dev_dataloader, optimizer, scheduler, tokenizer):
    model.train()
    global_step = 0
    wait_step = 0
    train_losses = []
    best_bleu4 = -1
    log_meteor = 0
    log_rouge_L = 0
    stop_training = False
    tensorboard_idx = 0

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    logger.info("Starting training!")
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for batch in epoch_iterator:

            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]

            # gen_loss + copyer_loss
            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3], input_node_ids=batch[4],
                         input_edge_ids=batch[5], node_length=batch[6], edge_length=batch[7], adj_matrix=batch[8],
                         is_training=True, use_copyer = True)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            # 写进训练损失
            writer.add_scalar('train/loss', loss, global_step)
            train_losses.append(loss.detach().cpu())
            loss.backward()

            # Gradient accumulation
            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            # Print loss and evaluate on the valid set
            if global_step % args.eval_period == 0:
                model.eval()
                tensorboard_idx += 1
                # ["Bleu_1","Bleu_2","Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]
                scores = inference(model if args.n_gpu == 1 else model.module, dev_dataloader, tokenizer, args, logger)
                logger.info("Step: %d Bleu-4: %.3f METEOR: %.3f ROUGE_L: %.3f on Epoch %d," % (global_step, scores['Bleu_4'] * 100.0, scores['METEOR'] * 100.0, scores['ROUGE_L'] * 100.0, epoch))
                curr_bleu = scores['Bleu_4']
                writer.add_scalars('metrics', {
                    'bleu4': scores['Bleu_4'],
                    'rougeL': scores['ROUGE_L'],
                    'meteor': scores['METEOR']
                }, tensorboard_idx)
                train_losses = []
                if best_bleu4 < curr_bleu:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(args.output_dir)
                    logger.info("Saving model with best %s: %.2f%% -> %.2f%% on epoch=%d, global_step=%d" %
                                ('BLEU-4', best_bleu4 * 100.0, curr_bleu * 100.0, epoch,
                                 global_step))
                    log_meteor = scores['METEOR'] * 100
                    log_rouge_L = scores['ROUGE_L'] * 100
                    best_bleu4 = curr_bleu
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break
                logger.info("Best BLEU-4: %.3f Best METEOR: %.3f Best ROUGE_L: %.3f " % (best_bleu4 * 100.0, log_meteor, log_rouge_L))
                
                # logger.info("best BLEU-4:" + str(best_bleu4) + "best METEOR:" + str(log_meteor) + "best ROUGE_L:" + str(log_rouge_L))
                model.train()
        if stop_training:
            break
    logger.info("Best BLEU_4 is:" + str(best_bleu4 * 100.0))


def inference(model, dev_dataloader, tokenizer, args, logger, save_predictions=False):
    predictions = []
    # Inference on the test set
    epoch_iterator = tqdm(dev_dataloader, desc="Test_Iteration")
    for i, batch in enumerate(epoch_iterator):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch]
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 input_node_ids=batch[4],
                                 input_edge_ids=batch[5],
                                 node_length=batch[6],
                                 edge_length=batch[7],
                                 adj_matrix=batch[8],
                                 num_beams=args.num_beams,
                                 length_penalty=args.length_penalty,
                                 max_length=args.max_output_length,
                                 early_stopping=True, )
        # Convert ids to tokens
        for input_, output in zip(batch[0], outputs):
            pred = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=args.clean_up_spaces)
            predictions.append(pred.strip())

    # Save the generated results
    if save_predictions:
        save_path = os.path.join(args.output_dir, "{}predictions.txt".format(args.prefix))
        with open(save_path, "w") as f:
            for pred in predictions:
                f.write(pred + '\n')
        logger.info("Saved prediction in {}".format(save_path))

    # data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
    # assert len(predictions) == len(data_ref)
    # return evaluate_bleu(data_ref=data_ref, data_sys=predictions)

    if 'webnlg' in args.dataset:
        data_ref = [data_ele['text'] for data_ele in dev_dataloader.dataset.data]
        assert len(predictions) == len(data_ref)
        return evaluate_bleu(data_ref=data_ref, data_sys=predictions)
    else:
        src_file = "./data/" + args.dataset + "/src-test.txt"
        tgt_file = "./data/" + args.dataset + "/tgt-test.txt"
        return eval(out_file=predictions, src_file=src_file, tgt_file=tgt_file)
