# Python wrapper for METEOR implementation, by Xinlei Chen --
# https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor.py
# Acknowledge Michael Denkowski for the generous discussion and help

import atexit
import logging
import os
import re
import subprocess
import sys
import threading
import psutil
from typing import List, Union
from .base_metric import BaseMetric

class METEORMetric(BaseMetric):
    def __init__(self, meteor_jar_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "meteor-1.5.jar"), **kwargs):
        """
        METEOR metric
            Taken from nlg-eval:
                # Python wrapper for METEOR implementation, by Xinlei Chen --
                # https://github.com/Maluuba/nlg-eval/blob/master/nlgeval/pycocoevalcap/meteor/meteor.py
                # Acknowledge Michael Denkowski for the generous discussion and help

            NOTE: assumes the presence of data/paraphrase-en.gz
            :param meteor_jar_path: location of METEOR jar
        """
        self.meteor_jar_path = meteor_jar_path
        # Used to guarantee thread safety
        self.lock = threading.Lock()
        mem = '2G'
        mem_available_gb = psutil.virtual_memory().available / 1E9
        if mem_available_gb < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), self.meteor_jar_path,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)
        
    def enc(self, s):
        return s.encode('utf-8')

    def dec(self, s):
        return s.decode('utf-8')

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)
            
    def __del__(self):
        self.close()

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        score_line = re.sub(r'\s+', ' ', score_line)
        self.meteor_p.stdin.write(self.enc(score_line))
        self.meteor_p.stdin.write(self.enc('\n'))
        self.meteor_p.stdin.flush()
        return self.dec(self.meteor_p.stdout.readline()).strip()
    
    def score(self, predictions: List[str], references: Union[None, List[List[str]]]=None, sources: Union[None, List[str]]=None) -> List[float]:
        segment_scores = []
        eval_line = 'EVAL'

        with self.lock:
            for pred, refs in zip(predictions, references):
                if not isinstance(refs, list):
                    refs = [refs]
                stat = self._stat(pred, refs)
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(self.enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()

            for _ in range(len(predictions)):
                v = self.meteor_p.stdout.readline()
                try:
                    segment_scores.append(float(self.dec(v.strip())))
                except Exception:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(self.dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))

        return segment_scores
