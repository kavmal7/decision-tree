import numpy as np
import pandas as pd
import decision_tree_id3

class RandomForestManual:
    def __init__(self, n_trees, max_features=None, impurity_measure='entropy'):
        self.n_trees = n_trees
        self.max_features = max_features
        self.impurity_measure = impurity_measure
        self.trees = []

    def bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap(X, y)
            tree = decision_tree_id3.id3(X_sample, y_sample, self.impurity_measure, self.max_features)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([self.predict_tree(tree, X) for tree in self.trees])
        final_preds = [self.majority_vote(row) for row in preds.T]
        return final_preds
    
    def predict_tree(self, tree, X):
        return np.array([decision_tree_id3.predict(x, tree) for x in X])
    
    def majority_vote(self, preds):
        vals, counts = np.unique(preds, return_counts=True)
        return vals[np.argmax(counts)]
