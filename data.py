import json
import copy
import random
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from eval_webnlg.measure_scores import run_coco_eval


# 计算评价指标
def evaluate_bleu(data_ref, data_sys):
    coco_eval = run_coco_eval(data_ref, data_sys)
    scores = {metric: score for metric, score in list(coco_eval.eval.items())}
    return scores


# ["Bleu_1","Bleu_2","Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr"]


# DataLoader
class WebNLGDataLoader(DataLoader):
    def __init__(self, args, dataset, mode):
        if mode == "train":
            sampler = RandomSampler(dataset)
            batch_size = args.train_batch_size
        else:
            sampler = SequentialSampler(dataset)
            batch_size = args.predict_batch_size
        super(WebNLGDataLoader, self).__init__(dataset, sampler=sampler, batch_size=batch_size,
                                               num_workers=args.num_workers)


class WebNLGDataset(Dataset):
    def __init__(self, logger, args, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.args = args
        with open(self.data_path + '.json', 'r') as f:
            self.data = json.load(f)
        logger.info("Total samples = {}".format(len(self.data)))
        if args.debug:
            self.data = self.data[:1000]
        assert type(self.data) == list
        assert all(["id" in d for d in self.data]), self.data[0].keys()
        if type(self.data[0]["id"]) == int:
            for i in range(len(self.data)):
                self.data[i]["id"] = str(self.data[i]["id"])

        self.head_ids, self.rel_ids, self.tail_ids = self.tokenizer.encode(' [head]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [relation]', add_special_tokens=False), \
                                                     self.tokenizer.encode(' [tail]', add_special_tokens=False)
        self.graph_ids, self.text_ids = self.tokenizer.encode(' [graph]', add_special_tokens=False), \
                                        self.tokenizer.encode(' [text]', add_special_tokens=False)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        if self.args.append_another_bos:
            self.add_bos_id = [self.tokenizer.bos_token_id] * 2
        else:
            self.add_bos_id = [self.tokenizer.bos_token_id]

    def __len__(self):
        return len(self.data)

    def get_all_entities_per_sample(self, mark_entity_number, mark_entity, entry):
        text_entity = set()
        text_relation = set()
        for entity_id in mark_entity_number:
            entity = entry['kbs'][entity_id]
            if len(entity[0]) == 0:
                continue
            for rel in entity[2]:
                if len(rel[0]) != 0 and len(rel[1]) != 0:
                    text_relation.add(rel[0])
                    text_entity.add(rel[1])

        text_entity_list = list(text_entity)
        text_relation_list = list(text_relation)
        for entity_ele in mark_entity:
            if entity_ele in text_entity_list:
                text_entity_list.remove(entity_ele)

        return text_entity_list, text_relation_list

    def get_change_per_sample(self, mark_entity, text_entity, text_relation):
        # during fine-tuning, we don't mask entities or relations
        ent_change = {}
        total_entity = mark_entity + text_entity

        for ent_id in range(len(total_entity)):
            entity_toks = self.tokenizer.encode(" {}".format(total_entity[ent_id]), add_special_tokens=False)
            ent_change[total_entity[ent_id]] = [entity_toks, ent_id]

        # relation change only includes the relation tokens and ids
        rel_change = {}
        for rel_id in range(len(text_relation)):
            rel_change[text_relation[rel_id]] = self.tokenizer.encode(' {}'.format(text_relation[rel_id]),
                                                                      add_special_tokens=False)

        return ent_change, rel_change

    def linearize_v2(self, entity, entity_change, head_ids, rel_ids, tail_ids,
                     relation_change, cnt_edge, adj_matrix):
        # string_label: encoder ids
        # string_label_tokens: encoder tokens

        if len(entity[0]) == 0:
            return [], '', [], [], cnt_edge, adj_matrix
        nodes, edges = [], []
        string_label = copy.deepcopy(head_ids)
        string_label_tokens = ' [head]'
        nodes.extend([-1] * len(string_label))
        edges.extend([-1] * len(string_label))

        string_label += entity_change[entity[0]][0]
        string_label_tokens += ' {}'.format(entity[0])
        nodes.extend([entity_change[entity[0]][1]] * len(entity_change[entity[0]][0]))
        edges.extend([-1] * len(entity_change[entity[0]][0]))

        for rel in entity[2]:
            if len(rel[0]) != 0 and len(rel[1]) != 0:
                rel_label = relation_change[rel[0]]
                rel_label_token = copy.deepcopy(rel[0])
                words_label = rel_ids + rel_label + tail_ids + entity_change[rel[1]][0]
                words_label_tokens = ' [relation] {} [tail] {}'.format(rel_label_token, rel[1])
                nodes.extend(
                    [-1] * (len(rel_ids) + len(rel_label) + len(tail_ids)) + [entity_change[rel[1]][1]] * len(
                        entity_change[rel[1]][0]))
                edges.extend([-1] * len(rel_ids) + [cnt_edge] * len(rel_label) + [-1] * (
                        len(tail_ids) + len(entity_change[rel[1]][0])))
                if entity_change[entity[0]][1] < len(adj_matrix) and entity_change[rel[1]][1] < len(adj_matrix):
                    adj_matrix[entity_change[entity[0]][1]][entity_change[rel[1]][1]] = cnt_edge

                cnt_edge += 1
                string_label += words_label
                string_label_tokens += words_label_tokens

        assert len(string_label) == len(nodes) == len(edges)

        return string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix

    def truncate_pair_ar(self, a, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        length_a_b = self.args.max_input_length - len(add_bos_id) - len(graph_ids) - len(text_ids) - 1
        if len(a) > length_a_b:
            a = a[:length_a_b]
            node_ids = node_ids[:length_a_b]
            edge_ids = edge_ids[:length_a_b]
        input_ids = add_bos_id + graph_ids + a + text_ids + [self.tokenizer.eos_token_id]
        input_node_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + node_ids + [-1] * (len(text_ids) + 1)
        input_edge_ids = [-1] * (len(add_bos_id) + len(graph_ids)) + edge_ids + [-1] * (len(text_ids) + 1)
        attn_mask = [1] * len(input_ids) + [0] * (self.args.max_input_length - len(input_ids))

        input_ids += [self.tokenizer.pad_token_id] * (self.args.max_input_length - len(input_ids))

        input_node_ids += [-1] * (self.args.max_input_length - len(input_node_ids))
        input_edge_ids += [-1] * (self.args.max_input_length - len(input_edge_ids))
        assert len(input_ids) == len(attn_mask) == self.args.max_input_length == len(input_node_ids) == len(
            input_edge_ids)
        return input_ids, attn_mask, input_node_ids, input_edge_ids

    def ar_prep_data(self, answers, questions, add_bos_id, graph_ids, text_ids, node_ids, edge_ids):
        # add bos and eos
        decoder_label_ids = copy.deepcopy(answers)

        if len(decoder_label_ids) > self.args.max_output_length - len(add_bos_id) - 1:
            decoder_label_ids = decoder_label_ids[:(self.args.max_output_length - len(add_bos_id) - 1)]

        decoder_label_ids = add_bos_id + decoder_label_ids + [self.tokenizer.eos_token_id]
        decoder_attn_mask = [1] * len(decoder_label_ids) + [0] * (self.args.max_output_length - len(decoder_label_ids))
        decoder_label_ids += [self.tokenizer.pad_token_id] * (self.args.max_output_length - len(decoder_label_ids))
        assert len(decoder_label_ids) == self.args.max_output_length == len(decoder_attn_mask)

        input_ids, input_attn_mask, input_node_ids, input_edge_ids = self.truncate_pair_ar(questions, add_bos_id,
                                                                                           graph_ids, text_ids,
                                                                                           node_ids, edge_ids)

        return input_ids, input_attn_mask, decoder_label_ids, decoder_attn_mask, input_node_ids, input_edge_ids

    def __getitem__(self, idx):

        # 取出一个样本，也就是一个小图
        obj = self.data[idx]
        entities = []
        # 将多个三元组放进entities里
        for _ in obj['kbs']:
            entities.append(_)
        strings_label = []
        strings_label_tokens = ''
        node_ids = []
        edge_ids = []
        cnt_edge = 0
        mark_entity = [obj['kbs'][ele_entity][0] for ele_entity in entities]
        mark_entity_number = entities
        text_entity, text_relation = self.get_all_entities_per_sample(mark_entity_number, mark_entity, obj)
        entity_change, relation_change = self.get_change_per_sample(mark_entity, text_entity, text_relation)
        adj_matrix = [[-1] * (self.args.max_node_length + 1) for _ in range(self.args.max_node_length + 1)]
        for i, entity_id in enumerate(entities):
            entity = obj['kbs'][entity_id]
            string_label, string_label_tokens, nodes, edges, cnt_edge, adj_matrix = self.linearize_v2(
                entity,
                entity_change,
                self.head_ids,
                self.rel_ids, self.tail_ids,
                relation_change, cnt_edge, adj_matrix)
            strings_label += string_label
            strings_label_tokens += string_label_tokens
            node_ids += nodes
            edge_ids += edges
        words_label_ids, words_label_tokens = [], ''
        current_text = random.choice(obj['text'])
        for word in current_text.split():
            word_label_ids = self.tokenizer.encode(" {}".format(word), add_special_tokens=False)
            word_label_tokens = copy.deepcopy(word)
            words_label_ids += word_label_ids
            words_label_tokens += ' ' + word_label_tokens
        input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, input_node_ids_ar, input_edge_ids_ar = \
            self.ar_prep_data(words_label_ids, strings_label, self.add_bos_id, self.graph_ids,
                              self.text_ids, node_ids, edge_ids)

        

        node_length_ar = max(input_node_ids_ar) + 1
        edge_length_ar = max(input_edge_ids_ar) + 1

        # print(node_length_ar)

        def masked_fill(src, masked_value, fill_value):
            return [src[src_id] if src[src_id] != masked_value and src[src_id] < fill_value else fill_value for src_id
                    in range(len(src))]

        input_node_ids_ar, input_edge_ids_ar = masked_fill(input_node_ids_ar, -1, self.args.max_node_length), \
                                               masked_fill(input_edge_ids_ar, -1, self.args.max_edge_length)

        def masked_fill_matrix(adj_matrix_input, masked_value, fill_value):
            adj_matrix_tmp = copy.deepcopy(adj_matrix_input)
            for a_id in range(len(adj_matrix_tmp)):
                for b_id in range(len(adj_matrix_tmp)):
                    if adj_matrix_tmp[a_id][b_id] == masked_value or adj_matrix_tmp[a_id][b_id] > fill_value:
                        adj_matrix_tmp[a_id][b_id] = fill_value
            return adj_matrix_tmp

        adj_matrix_ar = masked_fill_matrix(adj_matrix, -1, self.args.max_edge_length)

        assert len(input_ids_ar) == len(attn_mask_ar) == self.args.max_input_length == len(input_node_ids_ar) == len(
            input_edge_ids_ar)
        assert len(decoder_label_ids) == len(decoder_attn_mask) == self.args.max_output_length

        input_ids_ar = torch.LongTensor(input_ids_ar)
        attn_mask_ar = torch.LongTensor(attn_mask_ar)
        decoder_label_ids = torch.LongTensor(decoder_label_ids)
        decoder_attn_mask = torch.LongTensor(decoder_attn_mask)
        input_node_ids_ar = torch.LongTensor(input_node_ids_ar)
        input_edge_ids_ar = torch.LongTensor(input_edge_ids_ar)
        node_length_ar = torch.LongTensor([node_length_ar])
        edge_length_ar = torch.LongTensor([edge_length_ar])
        adj_matrix_ar = torch.LongTensor(adj_matrix_ar)

        return input_ids_ar, attn_mask_ar, decoder_label_ids, decoder_attn_mask, \
               input_node_ids_ar, input_edge_ids_ar, node_length_ar, edge_length_ar, adj_matrix_ar
