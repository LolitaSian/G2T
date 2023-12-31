#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

import os
import sys
import subprocess
import tempfile

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'eval_wqpq/meteor/meteor-1.5.jar'
#METEOR_JAR = 'core/evaluation/meteor/meteor-1.5.jar'

# java -Xmx2G -jar  core/evaluation/meteor/meteor-1.5.jar pred reference -l en -norm

class Meteor:

    def __init__(self):
        self.meteor_cmd = ' '.join(['java', '-Xmx2G', '-jar', METEOR_JAR, \
        '{pred}', '{reference}', '-l', 'en', '-norm'])


    def compute_score(self, gts, res):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        # Clean up a NamedTemporaryFile on your own
        # delete=True means the file will be deleted on close
        pred_tmp = tempfile.NamedTemporaryFile(mode='w', dir='./', delete=True)
        ref_tmp = tempfile.NamedTemporaryFile(mode='w', dir='./', delete=True)
        for i in imgIds:
            assert(len(res[i]) == 1) # only one prediction per example
            # do stuff with temp
            pred_tmp.write('{}\n'.format(res[i][0]))
            ref_tmp.write('{}\n'.format(gts[i][0]))

        pred_tmp.flush()
        ref_tmp.flush()

        output = subprocess.getoutput(self.meteor_cmd.format(pred=pred_tmp.name, reference=ref_tmp.name))
        #print(output)
        score = float(output.split('\n')[-1].split(':')[-1].strip())
        pred_tmp.close()  # deletes the file
        ref_tmp.close()  # deletes the file

        return score, None

    def method(self):
        return "METEOR"

