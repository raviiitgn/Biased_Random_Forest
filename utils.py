import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load a CSV file
def load_csv(file_path="./data/diabetes.csv"):
    return pd.read_csv(file_path)


# based on Scikit learn implementation
def stratified_k_fold(x, y, n_folds=10, shuffle=True, seed=100):
    """
    :param x: Not required
    :param y: class vector
    :param n_folds: desired folds
    :param shuffle: shuffling samples
    :param seed:
    :return: test index partition
    """
    for f in range(n_folds):
        shape = np.shape(y)
        if len(shape) == 1:
            y = np.ravel(y)
        elif len(shape) == 2 and shape[1] == 1:
            y = np.ravel(y)
        np.random.seed(seed)
        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, c_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = c_perm[y_inv]

        n_classes = len(y_idx)
        y_order = np.sort(y_encoded)
        allocation = np.asarray([np.bincount(y_order[i::n_folds], minlength=n_classes) for i in range(n_folds)])

        test_folds = np.empty(len(y), dtype='i')
        for k in range(n_classes):
            folds_for_class = np.arange(n_folds).repeat(allocation[:, k])
            if shuffle:
                np.random.shuffle(folds_for_class)

            test_folds[y_encoded == k] = folds_for_class

    return test_folds


def euclidean_distance(v1,v2):
    return np.sqrt(np.sum((v1 - v2)**2))

def euclidean_distance_matrix(data):
    dist_array = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(int(len(data)/2)):
            if i==j:
                dist_array[i][j] = np.inf
            else:
                dist_array[i][j] = euclidean_distance(data[i], data[j])

    return dist_array
#Taken from https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
def np_ffill(arr, axis=0):
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim==i else np.newaxis
        for dim in range(len(arr.shape))])]
        for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

#Precision = TruePositives / (TruePositives + FalsePositives)
#Recall = TruePositives / (TruePositives + FalseNegatives)

#Note, throws warning for possible divisions by zero

def precision_recall(preds, labels):
    precision = []
    recall = []
    ixs = np.arange(0, 1, 0.01)
    for i in ixs:
        preds_dummy = np.zeros_like(preds)
        preds_dummy[preds >= i] = 1

        tp = np.sum((preds_dummy == labels) & (preds_dummy == 1))
        fp = np.sum((preds_dummy == 1) & (labels == 0))
        pr = tp / (tp + fp)

        fn = np.sum((preds_dummy == 0) & (labels == 1))
        re = tp / (tp + fn)

        precision.append(pr)
        recall.append(re)

    recall = np.array(recall)
    recall = np_ffill(recall)

    precision = np.array(precision)
    precision = np_ffill(precision)

    area_under = np.trapz(x=recall, y=precision)
    return precision, recall, area_under


def tpr_fpr(preds, labels):
    true_pr = []
    false_pr = []
    n = len(labels)
    ixs = np.arange(0, 1, 0.01)

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)
    for i in ixs:
        preds_dummy = np.zeros_like(preds)
        preds_dummy[preds >= i] = 1

        tpr = np.sum((preds_dummy == labels) & (preds_dummy == 1)) / P
        fpr = np.sum((preds_dummy == 1) & (labels == 0)) / N

        true_pr.append(tpr)
        false_pr.append(fpr)

    true_pr = np.array(true_pr)
    false_pr = np.array(false_pr)
    true_pr = np_ffill(true_pr)
    false_pr = np_ffill((false_pr))

    area_under = np.trapz(x=false_pr, y=true_pr)
    return true_pr, false_pr, area_under
# Accuracy
def accuracy(preds, labels):
    return 100 * np.sum(preds == labels) / len(labels)

def std_agg(cnt, s1, s2):
    """
    :param cnt: number of samples (so far)
    :param s1: sum of squared values
    :param s2: sum of values
    :return: standard deviation
    """
    return np.sqrt((s2/cnt) - (s1/cnt)**2)

def entropy(data):
    """
    Calculate empirical entropy
    """
    if len(data)==1:
        return 0
    else:
        ent = stats.entropy(data, base=2)  # get entropy from counts
        return ent

class DecisionTree(object):
    def __init__(self, x, y, n_features, rand_cols, idxs, depth=10, min_leaf=5,criterion='gini'):
        """
        As defined in the RF class
        :param x: training samples
        :param y: target values
        :param n_features: number of features to consider
        :param f_idxs:
        :param idxs:
        :param depth:
        :param min_leaf:
        """

        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.rand_cols = rand_cols
        self.depth = depth
        self.n_features = n_features
        self.n, self.c = len(idxs), x.shape[1]
        self.val = np.mean(y[idxs])
        self.score = np.inf
        self.criterion = criterion
        self.trace = []
        self.find_varsplit()

    def find_varsplit(self):
        for i in self.rand_cols:
            self.find_better_split(i)

        if self.is_leaf():
            return

        x = self.split_cols_rows()

        #split the data in two subsets according to the best split point
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x > self.split)[0]

        #Randomly pick columns again
        lf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        rf_idxs = np.random.permutation(self.x.shape[1])[:self.n_features]
        #rand_idxs = np.random.permutation(self.input_shape[0])[:self.effective_sample_size]

        self.lhs = DecisionTree(self.x, self.y, self.n_features, lf_idxs, self.idxs[lhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf,criterion=self.criterion)
        self.rhs = DecisionTree(self.x, self.y, self.n_features, rf_idxs, self.idxs[rhs], depth=self.depth - 1,
                                min_leaf=self.min_leaf,criterion=self.criterion)

    # See: http://mystatisticsblog.blogspot.com/2017/12/building-random-forest-classifier-from.html
    # for the derivation of IG and entropy
    def find_better_split(self, var_idx):
        # Getting a column and the subsampled rows
        x_col, y_vals = self.x[self.idxs, var_idx], self.y[self.idxs]

        #Getting the indexes to order x (x is a column)
        #sorted only once by using argsort
        sort_idx = np.argsort(x_col)
        sort_x, sort_y = x_col[sort_idx], y_vals[sort_idx]
        #From zero up to n minus the minium leaf size to explor
        if self.criterion == 'gini':
            # left and right counters
            # measuring the value of y as the split point is traversed
            rhs_count, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y ** 2).sum()
            lhs_count, lhs_sum, lhs_sum2 = 0, 0, 0

            for i in range(0, self.n - self.min_leaf - 1):
                xi, yi = sort_x[i], sort_y[i]

                lhs_count += 1
                rhs_count -= 1
                lhs_sum += yi
                rhs_sum -= yi
                lhs_sum2 += yi ** 2
                rhs_sum2 -= yi ** 2

                # Stop criterion, if the remaining samples need to stay in the node
                # as min leaf has been reached
                # or current x value is the same than next (and hence all the ones that follow higher than)
                if (i < self.min_leaf) or (xi == sort_x[i + 1]):
                    continue

                lhs_std = std_agg(lhs_count, lhs_sum, lhs_sum2)
                rhs_std = std_agg(rhs_count, rhs_sum, rhs_sum2)

                #Summing up the weighted variance of the left and the right nodes
                curr_score = lhs_count * lhs_std + rhs_count * rhs_std
                if curr_score < self.score:
                    self.trace.append(self.score)
                    self.var_idx, self.score, self.split = var_idx, curr_score, xi
        elif self.criterion=="entropy":
            rhs_count = self.n
            lhs_count = 0
            for i in range(0, self.n - self.min_leaf - 1):
                lhs_count += 1
                rhs_count -= 1

                # See above
                if (i < self.min_leaf) or (sort_x[i] == sort_x[i + 1]):
                    continue

                lhs_ent = entropy(sort_y[:lhs_count])
                rhs_ent = entropy(sort_y[lhs_count:])

                # Summing up the weighted variance of the left and the right nodes
                curr_score = lhs_count * lhs_ent + rhs_count * rhs_ent

                if curr_score < self.score:
                    self.trace.append(self.score)
                    self.var_idx, self.score, self.split = var_idx, curr_score, sort_x[i]
        else:
            raise Exception("{} not defined".format(self.criterion))

    #@property
    def split_cols(self):
        return self.x[:,self.var_idx]

    #@property
    def split_cols_rows(self):
        return self.x[self.idxs, self.var_idx]

    #@property
    def is_leaf(self):
        return self.score == float('inf') or self.depth <= 0

    def predict(self, x):
        return np.array([self.predict_row(x_row) for x_row in x])

    def predict_row(self, xi):
        if self.is_leaf():
            return self.val

        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)


# Based on this RF implementation with some modifications to handle data as np arrays
# and allow split criterions [gini and entropy]
#https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249
class RandomForest(object):
    def __init__(self, n_trees=100, n_features=0.8, sample_size=0.8, criterion='gini', max_depth=6, min_leaf=5, seed=12):
        """
        :param n_trees: number of decision trees to generate
        :param n_features: umber of features to select at random for each decision tree.
            default 0.8 (defined after calling Fit)
            can also be 'sqrt'or 'log2' as commonly used
        :param sample_size: number of samples to take for each tree
        :param max_depth: maximum number of levels for the decision trees
        :param min_leaf: minimum number of samples per leaf
        :param seed: random seed (for reproducibility)
        """
        self.n_trees = n_trees
        self.n_features = n_features
        self.sample_size = sample_size
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed
        np.random.seed(seed)
        self.input_shape = None
            
    def create_tree(self):
        rand_idxs = np.random.permutation(self.input_shape[0])[:self.effective_sample_size]
        rand_cols = np.random.permutation(self.input_shape[1])[:self.effective_n_features]
        
        return DecisionTree(self.x[rand_idxs], self.y[rand_idxs], self.effective_n_features, rand_cols,
                            idxs=np.array(range(self.effective_sample_size)),
                            depth=self.max_depth, min_leaf=self.min_leaf, criterion=self.criterion)
    
    def fit(self, x,y):
        self.input_shape = x.shape
        if self.n_features == 'sqrt':
            self.effective_n_features = int(np.sqrt(self.input_shape[1]))
        elif self.n_features == 'log2':
            self.effective_n_features = int(np.log2(self.input_shape[1]))
        elif self.n_features is None:
            self.effective_n_features = self.input_shape[1]
        elif self.n_features < 1:
            self.effective_n_features = int(self.input_shape[1]*self.n_features)
        elif (self.n_features > 1 ) and (self.n_features < self.input_shape[1]):
            self.effective_n_features = self.input_shape[1]
        else:
            raise Exception("n_features is not defined correctly")

        if (self.sample_size > 0) and (self.sample_size < 1):
            self.effective_sample_size = int(self.input_shape[0]*self.sample_size)
        elif (self.sample_size>1) and (self.sample_size < self.input_shape[0]):
            self.effective_sample_size = int(self.sample_size)
        else:
            raise Exception("sample_size is not defined correctly")

        if type(x).__name__ == 'DataFrame':
            self._col_names = x.columns
            self.x = np.array(x)
        elif type(x).__name__ == 'ndarray':
            self._col_names = None
            self.x = x
        
        self.y = np.array(y)
        #Fitting a list of trees
        self.trees =[]
        for e, _ in enumerate(range(self.n_trees)):
            self.trees.append(self.create_tree())
            
    def predict(self,x):
        results = self.predict_proba(x)
        results[results>=0.5] = 1
        results[results<0.5] = 0

        return results

    def predict_proba(self, x,return_list=False):
        if return_list:
            return [t.predict(x) for t in self.trees]
        else:
            return np.mean([t.predict(x) for t in self.trees], axis=0)

class KNearestNeighbors(object):
    def __init__(self, k, distance='euclidean'):
        self.k = k
        self.distance = distance
        if self.distance == 'euclidean':
            self.dist_fun = euclidean_distance
        else:
            raise Exception("distance {} not implemented".format(distance))
    
    #Note
    # Text says to search for the NN in the dataset, but the formula (Algorithm 1) mentions the Tmaj 
    def predict(self,x, eval_ixs, test_ix):
        if type(x).__name__ == 'DataFrame':
            x = np.array(x)

        distances = []
        for i in eval_ixs:
            distance = self.dist_fun(x[i], x[test_ix])
            distances.append(distance)

        distances=np.array(distances)
        nn = eval_ixs[np.argsort(distances)[:self.k]]
        return nn
        

class BiasedRandomForest(object):
    #K= 10, p = 0.5, s=100
    #S = n_trees, p = partition, K = nearest_neighbor
    def __init__(self, n_trees=100, k=10, p=0.5, n_features=0.8, sample_size=0.8, criterion='gini', max_depth=6, min_leaf=5, seed=12, debug=True):
        """
        :param n_trees: number of decision trees to generate
        :param n_features: umber of features to select at random for each decision tree.
            default 0.8 (defined after calling Fit)
            can also be 'sqrt'or 'log2' as commonly used
        :param sample_size: number of samples to take for each tree
        :param max_depth: maximum number of levels for the decision trees
        :param min_leaf: minimum number of samples per leaf
        :param seed: random seed (for reproducibility)
        :param debug: debuggin the function
        """
        self.n_trees = n_trees
        self.k = k
        self.p = p
        self.n_features = n_features
        self.sample_size = sample_size
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.seed = seed
        self.debug = debug
        np.random.seed(self.seed)
        self.input_shape = None
        self.knn = KNearestNeighbors(k=self.k)

        self.x = None
        self.y = None

        self.min_class = None
        self.min_ixs = None
        self.maj_ixs = None
        self.critical_ixs = None

        self.n_trees_first = None
        self.n_trees_second = None

        self.first_rf = None
        self.second_rf = None
        
    def split_data(self):
        unique_vals, counts = np.unique(self.y, return_counts=True)
        min_class = unique_vals[np.argmin(counts)]
        self.min_class = min_class
        aux_bool = self.y==min_class
        self.min_ixs = np.arange(len(self.y))[aux_bool]
        self.maj_ixs = np.arange(len(self.y))[~aux_bool]
        
        self.critical_ixs = set()
        
        for i in self.min_ixs:
            self.critical_ixs.add(i)
            nn_ixs = self.knn.predict(self.x, self.maj_ixs, i)
            self.critical_ixs.update(nn_ixs) # Adding nn to critical set
            
        self.critical_ixs = list(self.critical_ixs)

        if self.debug:
            plt.scatter(self.x[:,0], self.x[:,1], c=self.y)
            plt.title("Whole dataset")
            plt.show()
            
            plt.scatter(self.x[self.critical_ixs,0], self.x[self.critical_ixs,1], c=self.y[self.critical_ixs])
            plt.title("Critical dataset")
            plt.show()
            
    def fit(self, x, y):
        self.x = x
        self.y = y
        
        self.split_data()
        
        self.n_trees_first = int(self.n_trees*(1-self.p))
        self.n_trees_second = int(self.n_trees*self.p)
        self.first_rf = RandomForest(self.n_trees_first, n_features=self.n_features, sample_size=self.sample_size,
                                     criterion=self.criterion, max_depth=self.max_depth, min_leaf=self.min_leaf)
        self.first_rf.fit(self.x, self.y)
        
        self.second_rf = RandomForest(self.n_trees_second, n_features=self.n_features, sample_size=self.sample_size,
                                      criterion=self.criterion, max_depth=self.max_depth, min_leaf=self.min_leaf)
        self.second_rf.fit(self.x[self.critical_ixs], self.y[self.critical_ixs])
    
    def predict(self,x):
        results = self.predict_proba(x)
        results[results>=0.5] = 1
        results[results<0.5] = 0

        return results
    
    def predict_proba(self,x):
        preds = self.first_rf.predict_proba(x, return_list=True)
        preds += self.second_rf.predict_proba(x, return_list=True)

        return np.mean(preds,axis=0)
