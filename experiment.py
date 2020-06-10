import numpy as np
import argparse
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file

sys.path.append('../')

from cs_ope_estimator import ope_estimators



def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Covariate Shift Adaptation for Off-policy Evaluation and Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='staimage',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    args = parser.parse_args(arguments)

    if args.preset == 'satimage':
        args.sample_size = 800
        args.dataset = 'satimage'
        args.num_trials = 100
    elif args.preset == 'vehicle':
        args.sample_size = 800
        args.dataset = 'vehicle'
        args.num_trials = 100
    elif args.preset == "pendigits":
        args.sample_size = 800
        args.dataset = 'pendigits'
        args.num_trials = 100
    return args

def data_generation(data_name, N):
    X, Y = load_svmlight_file('data/%s'%data_name)
    X = X.toarray()
    X = X/X.max(axis=0)
    Y = np.array(Y, np.int64)

    N_train = np.int(N*0.7)
    N_test= N - N_train

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1
    elif data_name == 'vehicle.scale':
        Y = Y - 1

    classes = np.unique(Y)

    Y_matrix = np.zeros(shape=(N, len(classes)))

    for i in range(N):
        Y_matrix[i, Y[i]] = 1
    
    return X, Y, Y_matrix, classes

def behavior_and_evaluation_policy(X, Y, classes, alpha=0.7):
    N = len(X)
    num_class = len(classes)
        
    classifier = LogisticRegression(random_state=0, penalty='l2', C=0.1, solver='saga', multi_class='multinomial',).fit(X, Y)
    predict = np.array(classifier.predict(X), np.int64)

    pi_predict = np.zeros(shape=(N, num_class))

    for i in range(N):
        pi_predict[i, predict[i]] = 1

    pi_random = np.random.uniform(size=(N, num_class))
    
    pi_random = pi_random.T
    pi_random /= pi_random.sum(axis=0)
    pi_random = pi_random.T
    
    pi_behavior = alpha*pi_predict + (1-alpha)*pi_random
        
    pi_evaluation = 0.9*pi_predict + 0.1*pi_random
        
    return pi_behavior, pi_evaluation

def true_value(Y_matrix, pi_evaluation, N):
     return np.sum(Y_matrix*pi_evaluation)/N
