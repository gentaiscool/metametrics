from base_task import BaseTask
from metametrics.optimizer.gp import GaussianProcessOptimizer

import logging
import os

logging.basicConfig(level=logging.INFO)

def ranking_acc(y_test, y_pred):
    correct_count = 0
    total_pairs = 0

    for i in range(0, len(y_pred), 2):
        # Compare the two consecutive rows
        if y_pred[i] > y_pred[i + 1] and y_test[i] > y_test[i + 1]:
            correct_count += 1
        elif y_pred[i] < y_pred[i + 1] and y_test[i] < y_test[i + 1]:
            correct_count += 1
        total_pairs += 1

    # Calculate the accuracy
    accuracy = correct_count / total_pairs
    return accuracy

class RewardBenchTask(BaseTask):
    def __init__(self, need_calibrate: bool=False):
        self.need_calibrate = need_calibrate
        self.metric_list = []
        self.optimizer = GaussianProcessOptimizer(ranking_acc) # By default
    
    def add_metric(self, metric):
        if metric not in self.metric_list:
            self.metric_list.append(metric)
            self.need_calibrate = True
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.need_calibrate = True
        
    def evaluate_metrics(self, dataset, **kwargs):
        ### PLACEHOLDER BEGIN ###
        # Placeholder to run everything; Need to clone rewardbench and install it
        def run_models(model_name, dataset, batch_size, output_dir):
            command = f"rewardbench --dataset {dataset} --model={model_name} --batch_size={batch_size} --save_all --output_dir {output_dir} --trust_remote_code"
            os.system(command)
            
        for metric in self.metric_list():
            model_name, output_dir, batch_size = metric.get('model_name'), metric.get('output_dir'), metric.get('batch_size', 1)
            run_models(model_name, dataset, batch_size, output_dir)
        ### PLACEHOLDER END ###
        
    def calibrate(self, X_train, Y_train, **kwargs):
        self.optimizer.optimize(X_train, Y_train, **kwargs)
        self.need_calibrate = False
    
    def evaluate_task(self, X_test, Y_test):
        if self.need_calibrate:
            logging.error("Modification to MetaMetrics was made, calibration is needed before evaluation!")
            raise RuntimeError()
        else:
            Y_pred = self.optimizer.predict(X_test)
            result = self.optimizer.evaluate(Y_pred, Y_test)
            return Y_pred, result
            
        
        
