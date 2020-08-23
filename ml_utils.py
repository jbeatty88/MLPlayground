import sklearn
from sklearn import linear_model
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import scale, MinMaxScaler, RobustScaler, Normalizer
import tensorflow.python as tf


def standardize_data(data):
    return scale(data)

def ranged_data(data, value_range):
  min_max_scaler = MinMaxScaler(feature_range=value_range)
  scaled_data = min_max_scaler.fit_transform(data)
  return scaled_data

def robust_scaling(data):
  robust_scaler = RobustScaler()
  scaled_data = robust_scaler.fit_transform(data)
  return scaled_data

def normalize_data(data):
  normalizer = Normalizer()
  norm_data = normalizer.fit_transform(data)
  return norm_data

def pca_data(data, n_components):
  pca_obj = PCA(n_components=n_components)
  component_data = pca_obj.fit_transform(data)
  return component_data

def linear_reg(data, labels):
  reg = sklearn.linear_model.LinearRegression()
  reg.fit(data, labels)
  return reg

def cv_ridge_reg(data, labels, alphas):
  # cv means cross-validation
  reg = linear_model.RidgeCV(alphas=alphas)
  reg.fit(data, labels)
  return reg

def lasso_reg(data, labels, alpha):
  reg = linear_model.Lasso(alpha=alpha)
  reg.fit(data, labels)
  return reg

def bayes_ridge(data, labels):
  reg = linear_model.BayesianRidge()
  reg.fit(data, labels)
  return reg

def multiclass_lr(data, labels, max_iter):
  reg = linear_model.LogisticRegression(solver='lbfgs', max_iter=max_iter, multi_class='multinomial')
  reg.fit(data, labels)
  return reg

def dataset_splitter(data, labels, test_size=0.25):
  split_dataset = train_test_split(data, labels, test_size=test_size)
  train_set = (split_dataset[0], split_dataset[2])
  test_set = (split_dataset[1], split_dataset[3])
  return (train_set, test_set)


def cv_decision_tree(is_clf, data, labels,
                     max_depth, cv):
  d_tree = sklearn.tree.DecisionTreeClassifier(max_depth=max_depth) if is_clf else sklearn.tree.DecisionTreeRegressor(
    max_depth=max_depth)
  scores = cross_val_score(d_tree, data, labels, cv=cv)
  return scores

def kmeans_clustering(data, n_clusters, batch_size):
  kmeans = KMeans(n_clusters=n_clusters) if batch_size == None else MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
  kmeans.fit(data)
  return kmeans

def init_inputs(input_size):
  inputs = tf.placeholder(tf.float32, shape=(None, input_size), name='inputs')
  return inputs

def init_labels(output_size):
  labels = tf.placeholder(tf.int32, shape=(None, output_size), name='labels')
  return labels

def model_layers(inputs, output_size):
  # Single layer perceptron whose output is logits.
  logits = tf.layers.dense(inputs, output_size, name='logits')
  return logits

def model_layers1(inputs, output_size):
  hidden1 = tf.layers.dense(inputs, 5,
                            activation=tf.nn.relu,
                            name='hidden1')
  logits = tf.layers.dense(hidden1, output_size,
                           name='logits')
  return logits

def model_layers2(inputs, output_size):
  hidden1 = tf.layers.dense(inputs, 5,
                            activation=tf.nn.relu,
                            name='hidden1')
  hidden2 = tf.layers.dense(hidden1, 5,
                            activation=tf.nn.relu,
                            name='hidden2')
  logits = tf.layers.dense(hidden2, output_size,
                           name='logits')
  return logits

