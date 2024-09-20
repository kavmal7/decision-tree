import numpy as np
from sklearn.model_selection import train_test_split

# Class for a tree node, its features, labels, and subtrees
class TreeNode:
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        self.label = None

# Learn a decision tree
def learn(X, y, impurity_measure='entropy', prune=False, pruning_size=0.2):
   
   # "Trick" to convert to NumPy to avoid bug
    training_data = np.column_stack((X, y))
    X = training_data[:, :-1]
    y = training_data[:, -1]

    # If we wish to prune, split data, train tree, and then prune
    if prune:
        training_data_X, pruning_data_X, training_data_y, pruning_data_y = train_test_split(X, y, test_size=pruning_size, random_state=20)
        pruning_data = np.column_stack((pruning_data_X, pruning_data_y))
        majority_label = most_common(training_data_y)
        
        tree = id3(training_data_X, training_data_y, impurity_measure)
        prune_tree(tree, pruning_data, tree, majority_label)
    else:
        # Build tree as normal
        tree = id3(X, y, impurity_measure)

    return tree

# Perform the ID3 algorithm
def id3(X, y, impurity_measure, max_features=None):
    node = TreeNode()
    unique_labels = np.unique(y)

    # If there is only one label
    if len(unique_labels) == 1:
        node.label = unique_labels[0]
        return node
    
    # If the features are identical
    if (X.min(axis=0) == X.max(axis=0)).all():
        node.label = most_common(y)
        return node
    
    # Else, find the split with most information gain
    gain, best_feature, best_val = best_split(X, y, impurity_measure, max_features)

    # If no gain, return node with the majority label
    if gain == 0:
        node.label = most_common(y)
        return node
    elif gain is None:
        raise Exception('Cannot split empty dataset')
    
    # Split data on best split and recurse
    left_data_X, left_data_y, right_data_X, right_data_y = splitter(X, y, best_feature, best_val)
    left_tree = id3(left_data_X, left_data_y, impurity_measure, max_features)
    right_tree = id3(right_data_X, right_data_y, impurity_measure, max_features)

    # Assign and attach subtrees on back-recursion
    node.split_feature = best_feature
    node.split_value = best_val
    node.left = left_tree
    node.right = right_tree

    return node

# Calculate the best split based on the best information gain
def best_split(X, y, impurity_measure, max_features=None):

    # If data is empty, return none
    if len(X) == 0 or len(y) == 0:
        return None, None, None
    
    # Set impurity function
    if impurity_measure == 'entropy':
        impurity_function = entropy
    elif impurity_measure == 'gini':
        impurity_function = gini
    else:
        raise Exception('Incorrect impurity measure selected')

    # Set initial values and parent impurity
    best_gain = 0
    best_feature = None
    best_val = None
    impurity_val = impurity_function(y)

    feature_count = np.shape(X)[1]
    features = np.arange(feature_count)

    if max_features is not None:
        features = np.random.choice(features, max_features, replace=False)

    # Iterate over every feature
    for feature in range(feature_count):
        feature_vals = np.unique(X[:, feature])
        # Iterate over every unique value of the feature
        for val in feature_vals:
            # Split data on current iteration of feature and value
            left_temp_X, left_temp_y, right_temp_X, right_temp_y = splitter(X, y, feature, val)

            # If either left or right data is empty, then skip
            if len(left_temp_X) == 0 or len(right_temp_X) == 0:
                continue

            # Calculate gain via weighted average using children impurities
            prob_left = len(left_temp_X) / len(X)
            gain = impurity_val - (prob_left * impurity_function(left_temp_y) + (1 - prob_left) * impurity_function(right_temp_y))

            # Update gain upon improvement
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_val = val
    return best_gain, best_feature, best_val

# Split data on a given feature and its corresponding value
def splitter(X, y, feature, val):
    left_indices = X[:, feature] < val
    right_indices = ~left_indices  

    left_X = X[left_indices]
    left_y = y[left_indices]

    right_X = X[right_indices]
    right_y = y[right_indices]

    return left_X, left_y, right_X, right_y

# Calculate entropy of labels
def entropy(y):
    unique_labels = np.unique(y)
    total = len(y)
    entropy_val = 0

    for label in unique_labels:
        count = np.count_nonzero(y == label)
        prob = count / total
        entropy_val -= prob * np.log2(prob)
    
    return entropy_val

# Calculate gini impurity of labels
def gini(y):
    unique_labels = np.unique(y)
    gini_val = 1
    for label in unique_labels:
        count = np.count_nonzero(y == label)
        p = count / len(y)
        gini_val -= p**2

    return gini_val
        
# Predict label given a data point
def predict(x, node):
    while node.label is None:
        if x[node.split_feature] < node.split_value:
            node = node.left
        else:
            node = node.right

    return node.label

# Perform reduced-error pruning
def prune_tree(tree, pruning_data, root, majority_label):
    if tree.label is not None:
        return tree
    
    # Split data and recurse to leaves (allows for post-order traversal up from leaves when pruning)
    if tree.left:
        tree.left = prune_tree(tree.left, pruning_data, root, majority_label)
    
    if tree.right:
        tree.right = prune_tree(tree.right, pruning_data, root, majority_label)

    # Calculate accuracy on the whole tree before pruning
    pre_pruning_acc = accuracy(pruning_data, root)
    pre_left = tree.left
    pre_right = tree.right
    pre_lab = tree.label

    # Prune and calculate accuracy on the whole tree again
    tree.left, tree.right = None, None
    tree.label = majority_label
    post_pruning_acc = accuracy(pruning_data, root)

    # If accuracy worsens, restore old subtrees, else, maintain change and return
    if post_pruning_acc < pre_pruning_acc:
        tree.left, tree.right = pre_left, pre_right
        tree.label = pre_lab
        
    return tree

# Calculate accuracy of data on a decision tree
def accuracy(data, tree):
    if len(data) == 0:
        return 0
    predictions = [predict(d[:-1], tree) for d in data]
    return np.mean(np.array(predictions) == data[:, -1])

# Helper function to get the most common label 
def most_common(y):
    lab, count = np.unique(y, return_counts=True)
    return lab[np.argmax(count)]
