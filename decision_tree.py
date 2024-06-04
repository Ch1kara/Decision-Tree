from cmath import inf
import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):

        # used for all other nodes, aka decision nodes
        self.feature = feature
        self.threshold = threshold
        self.left_child = left
        self.right_child = right
        self.info_gain_amt = info_gain

        # values are only given to leaf nodes at the bottom of the tree
        self.value = value

    def check_leaf_node(self):
        """Returns True if a node is a leaf node and False if it is not"""
        if self.value is not None:
            return True
        else:
            return False


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, num_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.num_features = num_features
        self.root = None

    @staticmethod
    def split(column, threshold):
        """Splits the data between the left and right child given a threshold"""
        df_left = np.where(column <= threshold)[0]
        df_right = np.where(column > threshold)[0]
        return df_left, df_right

    @staticmethod
    def entropy(y):
        """Takes in the labels of a decision node and calculate its entropy"""
        labels = np.unique(y)
        entropy = 0
        # formula for calculating entropy
        for label in labels:
            p = len(y[y == label]) / len(y)
            entropy += -p * np.log2(p)
        return entropy

    def info_gain(self, parent, column, threshold, mode='entropy'):
        """Calculate the amount of information gain for a given decision node"""
        e_parent = self.entropy(parent)
        l_child, r_child = self.split(column, threshold)

        # check if either children are null (split not needed and info_gain is zero)
        if len(l_child) == 0 or len(r_child) == 0:
            return 0

        # calculate weighted entropy of children using formula
        weighted_l, weighted_r = len(l_child) / len(parent), len(r_child) / len(parent)
        e_l_child, e_r_child = self.entropy(parent[l_child]), self.entropy(parent[r_child])
        avg_child_entropy = (weighted_l * e_l_child) + (weighted_r * e_r_child)

        # calculate info gain for the given node
        information_gain = 0
        if mode == 'entropy':
            # formula for information gain
            information_gain = e_parent - avg_child_entropy

        return information_gain

    def best_split(self, X, y, features):
        """Use information gain to determine the best feature to split"""
        best_info_gain = -float(inf)  # starting at lowest possible value so it will only increase
        split_ind, split_thresh = None, None
        # loop through the features
        for feature_ind in features:
            column = X[:, feature_ind]
            # unique values in a given column
            thresholds = np.unique(column)
            # loop through the unique values of this feature and calculate the current info gain
            for threshold in thresholds:
                curr_info_gain = self.info_gain(y, column, threshold, mode = 'entropy')
                if curr_info_gain > best_info_gain:
                    split_ind = feature_ind
                    split_thresh = threshold
                    best_info_gain = curr_info_gain
        return split_ind, split_thresh

    @staticmethod
    def leaf_value(y):
        """Takes in the given labels and returns the most common label as the label of the leaf node"""
        y = list(y)
        return max(y, key=y.count)

    def build_tree(self, X, y, curr_depth=0):
        """Recursively builds the decision tree"""
        num_samples, num_features = X.shape
        num_targets = len(np.unique(y))

        # base case stopping conditions
        if num_samples < self.min_samples_split or curr_depth >= self.max_depth or num_targets == 1:
            # if satisfied, returns the leaf node and its value meaning that the entire dataset is pure
            leaf_value = self.leaf_value(y)
            return Node(value=leaf_value)

        # randomly choose the features we want to consider based on self.num_features (reduce dimensionality)
        feature_indices = np.random.choice(num_features, self.num_features, replace=False)

        # determine the best splitting feature given info gain
        split_feat, split_thresh = self.best_split(X, y, feature_indices)

        # create left and right subtrees by recursively calling on the build_tree function
        l_child_ind, r_child_ind = self.split(X[:, split_feat], split_thresh)
        left_tree = self.build_tree(X[l_child_ind, :], y[l_child_ind], curr_depth+1)
        right_tree = self.build_tree(X[r_child_ind, :], y[r_child_ind], curr_depth+1)
        # returns the root node that connects both parts of the tree
        return Node(split_feat, split_thresh, left_tree, right_tree)

    def fit(self, X, y):
        """Fit and train the tree based on the given data"""
        # Automatically makes the # of features we want to consider the max
        if not self.num_features:
            self.num_features = X.shape[1]
        # in the case that we inputted a higher or lower # of considered features
        else:
            self.num_features = min(X.shape[1], self.num_features)

        self.root = self.build_tree(X, y)

    def predict_one_value(self, X, node):
        """Takes in a node and recursively traverses the tree to find the leaf nodes value"""
        # https://www.youtube.com/watch?v=NxEHSAfFlK8
        if node.check_leaf_node():
            return node.value

        if X[node.feature] <= node.threshold:
            return self.predict_one_value(X, node.left_child)
        else:
            return self.predict_one_value(X, node.right_child)

    def predict(self, X):
        """Use the tree to predict on a test dataset"""
        predictions = []
        for x in X:
            prediction = self.predict_one_value(x, self.root)
            predictions.append(prediction)
        # need to compare an array with an array for accuracy_score
        return np.array(predictions)

    def print_tree(self, root_node=None):
        """Takes in a root node and prints the decision tree built from the root node"""
        pass

