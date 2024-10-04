
from bayes_opt import BayesianOptimization
import logging

logging.basicConfig(level=logging.INFO)

from base_optimizer import BaseOptimizer

class GaussianProcessOptimizer(BaseOptimizer):
    def __init__(self, objective_fn):
        self.objective_fn = objective_fn

    def black_box_function(self, metric_scores, human_scores, metric_weights):
        """Calculate the objective function score."""
        final_metric_scores = [
            sum(metric_scores[metric_name][i] * metric_weights[metric_name] for metric_name in metric_scores)
            for i in range(len(human_scores))
        ]

        return self.objective_fn(human_scores, final_metric_scores)

    def run_bayesian_optimization(self, metric_scores, metrics_names, human_scores, init_points, n_iter, seed):
        """Run Bayesian optimization with given parameters."""
        def wrapped_black_box_function(**metric_weights):
            return self.black_box_function(metric_scores, human_scores, metric_weights)
        
        pbounds = {metric_name: (0, 1) for metric_name in metrics_names}
        optimizer = BayesianOptimization(f=wrapped_black_box_function, pbounds=pbounds, random_state=seed)
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        return optimizer

    def optimize(self, X_train, Y_train, init_points=5, n_iter=100, seed=1):
        """Optimize metrics weights based on given seed."""
        metrics_used = X_train.columns
        metric_scores = {}
        for i, metric_name in enumerate(metrics_used):
            metric_scores[metric_name] = X_train[metric_name].to_list()
        human_scores = Y_train.to_list()
        
        optimizer = self.run_bayesian_optimization(X_train, metric_scores, metrics_used, human_scores,
                                                   init_points, n_iter, seed)
        
        # Evaluate
        X_train['final_score'] = X_train.apply(
            lambda row: sum(row[col] * optimizer.max['params'].get(col, 0) for col in metrics_used), axis=1)
        kt = self.objective_fn(X_train["final_score"], Y_train)
        logging.debug(f"Correlation gained from the current optimizer is {kt.correlation} where metrics used are {metrics_used}")
        
        self.optimizer_weights = optimizer.max
    
    def predict(self, X_feats):
        metrics_used = X_feats.columns
        X_feats['final_score'] = X_feats.apply(
            lambda row: sum(row[col] * self.optimizer_weights['params'].get(col, 0) for col in metrics_used), axis=1)
        return X_feats['final_score'].to_numpy()
        
    def evaluate(self, Y_pred, Y_expected):
        return self.objective_fn(Y_pred, Y_expected)
