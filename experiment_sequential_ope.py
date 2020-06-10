import argparse
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
from sklearn.neighbors import KNeighborsRegressor
from nadaraya_watson import KernelRegression
from sklearn.decomposition import PCA


def process_args(arguments):
    parser = argparse.ArgumentParser(
        description='Off-policy Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', '-d', type=str, default='satimage',
                        help='Name of dataset')
    parser.add_argument('--sample_size', '-s', type=int, default=1000,
                        help='Sample size')
    parser.add_argument('--num_trials', '-n', type=int, default=200,
                        help='The number of trials')
    parser.add_argument('--preset', '-p', type=str, default=None,
                        choices=['satimage', 'vehicle', 'pendigits'],
                        help="Presets of configuration")
    parser.add_argument('--method', '-m', type=str, default=None,
                        choices=['NW_regression', 'knn_regression'])
    parser.add_argument('--ratio', '-r', type=float, default=0.7,
                        help="ratio")
    parser.add_argument('--para_r', '-pr', type=float, default=0.7,
                        help="ratio")
    parser.add_argument('--tau', '-tau', type=float, default=0.7,
                        help="tau")
    parser.add_argument('--gamma', '-g', type=float, default=0.,
                        help="gamma")
    parser.add_argument('--policy_type', '-t', type=str, default='UCB',
                        help="policy type")
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


class Basic(object):
    def __init__(self, num_arm, T):
        self.num_arm = num_arm
        self.sum_of_reward = np.zeros(num_arm)
        self.num_of_trial = np.zeros(num_arm)
        self.T = T

class UCB(Basic):
    def __init__(self, num_arm, T, dim, sigma2_0=1, sigma2=1, alpha=1):
        super().__init__(num_arm, T)
        self.ucb_score = np.zeros(num_arm)
        self.identity = np.identity(dim)
        
        self.sigma2_0 = sigma2_0
        self.sigma2 = sigma2
        
        self.A_inv_array = [(self.sigma2_0/self.sigma2)*self.identity for i in range(num_arm)]
        
        self.b_array = [np.zeros((dim, 1)) for i in range(num_arm)]
        
        self.alpha = alpha

    def __call__(self, t, covariate):
        alpha_t = self.alpha*np.sqrt(np.log(t+1))
        
        for arm in range(self.num_arm):
            theta = self.A_inv_array[arm].dot(self.b_array[arm])
            m0 = covariate.T.dot(theta)
            m1 = alpha_t*np.sqrt(self.sigma2)*np.sqrt(covariate.T.dot(self.A_inv_array[arm]).dot(covariate))
            self.ucb_score[arm] = m0 + m1
        
        return np.argmax(self.ucb_score)

    def update(self, arm, reward, covariate):
        self.sum_of_reward[arm] += reward
        self.num_of_trial[arm] += 1
        
        A_inv_temp = self.A_inv_array[arm].copy()
        A_inv_temp0 = A_inv_temp.dot(covariate).dot(covariate.T).dot(A_inv_temp)
        A_inv_temp1 = 1+covariate.T.dot(A_inv_temp).dot(covariate)
        self.A_inv_array[arm] -= A_inv_temp0/A_inv_temp1
        
        self.b_array[arm] += covariate*reward

class TS(Basic):
    def __init__(self, num_arm, T, dim, sigma2_0=1, sigma2=1, alpha=1):
        super().__init__(num_arm, T)
        self.ucb_score = np.zeros(num_arm)
        self.identity = np.identity(dim)
        
        self.sigma2_0 = sigma2_0
        self.sigma2 = sigma2
        
        self.A_inv_array = [(self.sigma2_0/self.sigma2)*self.identity for i in range(num_arm)]
        
        self.b_array = [np.zeros((dim, 1)) for i in range(num_arm)]
        
        self.alpha = alpha

    def __call__(self, t, covariate):
        
        for arm in range(self.num_arm):
            mu = self.A_inv_array[arm].dot(self.b_array[arm])
            
            theta = np.random.multivariate_normal(mu.T[0], self.sigma2*self.A_inv_array[arm])
            
            self.ucb_score[arm] = covariate.T.dot(theta)
        
        return np.argmax(self.ucb_score)

    def update(self, arm, reward, covariate):
        self.sum_of_reward[arm] += reward
        self.num_of_trial[arm] += 1
        
        A_inv_temp = self.A_inv_array[arm].copy()
        A_inv_temp0 = A_inv_temp.dot(covariate).dot(covariate.T).dot(A_inv_temp)
        A_inv_temp1 = 1+covariate.T.dot(A_inv_temp).dot(covariate)
        self.A_inv_array[arm] -= A_inv_temp0/A_inv_temp1
        
        self.b_array[arm] += covariate*reward

def data_generation(data_name, N):
    X, Y = load_svmlight_file('data/%s'%data_name)
    X = X.toarray()
    maxX = X.max(axis=0)
    maxX[maxX == 0] = 1
    X = X/maxX
    Y = np.array(Y, np.int64)

    perm = np.random.permutation(len(X))

    X, Y = X[perm[:N]], Y[perm[:N]]

    if data_name == 'satimage.scale':
        Y = Y - 1
    elif data_name == 'vehicle.scale':
        Y = Y - 1
    elif data_name == 'usps':
        Y = Y - 1
        pca = PCA(n_components=100).fit(X)
        X = pca.transform(X)
    elif data_name == 'mnist':
        pca = PCA(n_components=100).fit(X)
        X = pca.transform(X)
    elif data_name == 'dna.scale':
        Y = Y - 1
    elif data_name == 'shuttle.scale':
        Y = Y - 1
    elif data_name == 'letter.scale':
        Y = Y - 1
    elif data_name == 'covtype':
        Y = Y - 1
    elif data_name == 'Sensorless':
        Y = Y - 1

    action_sets = np.unique(Y)

    Y_matrix = np.zeros(shape=(N, len(action_sets)))

    for i in range(N):
        Y_matrix[i, Y[i]] = 1
    
    return X, Y_matrix, action_sets

def behavior_and_evaluation_policy(covariates, true_outcome_matrix, action_sets, alpha=0.7, policy_type='RW', predct_alg='Logit', tau=0.1):
    sample_size, dim = covariates.shape
    num_actions = len(action_sets)
    
    chosen_action_matrix = np.zeros(shape=(sample_size, len(action_sets)))
    observed_outcome_matrix = np.zeros(shape=(sample_size, len(action_sets)))
    
    if predct_alg == 'Logit':
        classifier = LogisticRegression(random_state=0, penalty='l2', C=0.1, solver='saga', multi_class='multinomial')
        
        chosen_action = np.zeros(sample_size)
        
        for i in range(sample_size):
            chosen_action[i] = np.where(true_outcome_matrix[i]==1)[0]
        
        classifier = classifier.fit(covariates, chosen_action)
        predict = np.array(classifier.predict(covariates), np.int64)

    pi_predict_matrix = np.zeros(shape=(sample_size, num_actions))

    for i in range(sample_size):
        pi_predict_matrix[i, predict[i]] = 1

    pi_random_matrix = np.random.uniform(size=(sample_size, num_actions))
    
    pi_random_matrix = pi_random_matrix.T
    pi_random_matrix /= pi_random_matrix.sum(axis=0)
    pi_random_matrix = pi_random_matrix.T
        
    pi_evaluation_matrix = 0.9*pi_predict_matrix + 0.1*pi_random_matrix
    
    print(tau)

    #random walk policy
    if policy_type == 'RW':
        pi_behavior_array = np.zeros((sample_size, sample_size, num_actions))
        next_candidate = np.random.uniform(0.01, 0.99, size=(1, num_actions))
        next_candidate = next_candidate/np.sum(next_candidate)
        pi_behavior_array[0] = next_candidate
        
        for time in range(1, sample_size):
            rand = 0.01*np.random.normal(0, 1, size=(1, num_actions))
            
            next_candidate = pi_behavior_array[time-1, time-1] + rand
            next_candidate[next_candidate < 0] = 0
            
            next_candidate = next_candidate/np.sum(next_candidate)
            
            uni_rand = np.random.uniform(size=(1, num_actions))
            uni_rand = uni_rand/np.sum(uni_rand)
            
            pi_behavior_array[time] = (1-tau)*uni_rand + tau*next_candidate

        for sample_idx in range(sample_size):
            chosen_action = np.random.choice(action_sets, p=pi_behavior_array[sample_idx, sample_idx])
            chosen_action_matrix[sample_idx, chosen_action] = 1
            observed_outcome_matrix[sample_idx, chosen_action] = true_outcome_matrix[sample_idx, chosen_action]
            
    if policy_type == 'UCB':
        pi_behavior_array = np.zeros((sample_size, sample_size, num_actions))
        next_candidate = np.random.uniform(0.01, 0.99, size=(1, num_actions))
        next_candidate = next_candidate/np.sum(next_candidate)
        pi_behavior_array[0] = next_candidate
        
        ucb = UCB(num_arm=num_actions, T=sample_size, dim=dim, sigma2_0=5, sigma2=5)

        for time in range(sample_size):
            covariate_t = np.array([covariates[time]]).T
            
            for time_temp in range(sample_size):
                covariate_t_temp = np.array([covariates[time_temp]]).T
                arm = ucb(time_temp, covariate_t_temp)
                uni_rand = np.random.uniform(size=(1, num_actions))
                uni_rand = uni_rand/np.sum(uni_rand)
                prob = (1-tau)*uni_rand
                prob[0, arm] += tau
                
                pi_behavior_array[time, time_temp] = prob
            
            chosen_action = np.random.choice(action_sets, p=pi_behavior_array[time, time])
            chosen_action_matrix[time, chosen_action] = 1
            observed_outcome_matrix[time, chosen_action] = true_outcome_matrix[time, chosen_action]
                
            ucb.update(chosen_action, true_outcome_matrix[time, chosen_action], covariate_t)
            
    if policy_type == 'TS':
        pi_behavior_array = np.zeros((sample_size, sample_size, num_actions))
        next_candidate = np.random.uniform(0.01, 0.99, size=(1, num_actions))
        next_candidate = next_candidate/np.sum(next_candidate)
        pi_behavior_array[0] = next_candidate
        
        ts = TS(num_arm=num_actions, T=sample_size, dim=dim, sigma2_0=1, sigma2=1)

        for time in range(sample_size):
            covariate_t = np.array([covariates[time]]).T
            
            for time_temp in range(sample_size):
                covariate_t_temp = np.array([covariates[time_temp]]).T
                arm = ts(time_temp, covariate_t_temp)
                uni_rand = np.random.uniform(size=(1, num_actions))
                uni_rand = uni_rand/np.sum(uni_rand)
                prob = (1-tau)*uni_rand
                prob[0, arm] += tau
                
                pi_behavior_array[time, time_temp] = prob
            
            chosen_action = np.random.choice(action_sets, p=pi_behavior_array[time, time])
            chosen_action_matrix[time, chosen_action] = 1
            observed_outcome_matrix[time, chosen_action] = true_outcome_matrix[time, chosen_action]
                
            ts.update(chosen_action, true_outcome_matrix[time, chosen_action], covariate_t)
            
    return pi_behavior_array, pi_evaluation_matrix, chosen_action_matrix, observed_outcome_matrix

def true_value(Y_matrix, pi_evaluation):
     return np.sum(Y_matrix*pi_evaluation)/Y_matrix.shape[0]

class OPE_Estimators():
    def __init__(self, covariates, chosen_action_matrix, observed_outcom_matrix, action_sets, pi_behavior_array, pi_evaluation_matrix):
        self.X = covariates
        self.A = chosen_action_matrix
        self.Y = observed_outcom_matrix
        self.action_sets = action_sets
        self.num_actions = len(action_sets)
        
        self.pi_behavior_array = pi_behavior_array
        self.pi_evaluation_matrix = pi_evaluation_matrix

        self.sample_size, self.dim = covariates.shape
        
        self.function_y_dict = {}
        self.function_y2_dict = {}

    def est_a3ipw_params(self, method='knn_regression'):
        
        if method not in self.function_y_dict.keys():
            self.construct_function_list(method)
         
    #estimate theta_0
    def est_a3ipw(self, para_r=0.1, ratio=0.1, gamma=0, method='knn_regression', epsilon=0.1):
        
        if method not in self.function_y_dict.keys():
            self.construct_function_list(method)
        
        #samples using for constructing the estimator
        rT = int(para_r*self.sample_size)
        gT =  int(gamma*self.sample_size)
        
        #load models
        mean_y_array_temp = self.function_y_dict[method]
        mean_y2_array_temp = self.function_y2_dict[method]
        
        #construct models for estimating theta
        mean_y_array = np.zeros((self.sample_size, self.num_actions))
        mean_y2_array = np.zeros((self.sample_size, self.num_actions))
        
        #using samples until (t-1)-period to predict the value of t-peirod
        for sample_idx in range(1, self.sample_size):
            mean_y_array[sample_idx] = mean_y_array_temp[sample_idx-1, sample_idx]
            mean_y2_array[sample_idx] = mean_y2_array_temp[sample_idx-1, sample_idx]
            
        #construct behavior policy for estimating theta
        pi_behavior_matrix = np.zeros((self.sample_size, self.num_actions))

        for sample_idx in range(self.sample_size):
            #probability at t-th period ditermined at t-th period (using t-1 samples)
            pi_behavior_matrix[sample_idx]  = self.pi_behavior_array[sample_idx, sample_idx]
                    
        #first stage estimation of theta
        self.theta_list = np.zeros(rT)
    
        for sample_idx in range(1, rT):
            temp_term = self.A[:sample_idx]*(self.Y[:sample_idx] - mean_y_array[:sample_idx])/pi_behavior_matrix[:sample_idx] + mean_y_array[:sample_idx]
            theta_temp = np.mean(np.sum(self.pi_evaluation_matrix[:sample_idx]*temp_term, axis=1))
            self.theta_list[sample_idx] = theta_temp
                            
        #first stage estimation of variance
        self.est_var_list = np.ones(rT)
        
        #\hat{var}_{t-1}
        var_temp_array = mean_y2_array_temp - mean_y_array_temp**2
        
        for sample_idx in range(rT):
            #\hat{E[y|x]}_{t-1}
            mean_temp = mean_y_array_temp[sample_idx, rT:]
            #\hat{Var[y|x]}_{t-1}
            var_temp = var_temp_array[sample_idx, rT:]
            var_temp[var_temp <= epsilon] = epsilon
            
            pi_temp = self.pi_behavior_array[sample_idx, rT:]
            
            temp_term1 = np.sum(self.pi_evaluation_matrix[rT:]**2*var_temp/pi_temp, axis=1)
            temp_term2 = (np.sum(self.pi_evaluation_matrix[rT:]*mean_temp, axis=1) - self.theta_list[sample_idx])**2

            self.est_var_list[sample_idx] = np.mean(temp_term1 + temp_term2)
        
        #second stage estimation
        temp_term = self.A[:rT]*(self.Y[:rT] - mean_y_array[:rT])/pi_behavior_matrix[:rT] + mean_y_array[:rT]
        theta_temp = np.sum(self.pi_evaluation_matrix[:rT]*temp_term, axis=1)
                        
        theta_a3ipw = np.sum(theta_temp/np.sqrt(self.est_var_list[:rT]))/(np.sum(1/np.sqrt(self.est_var_list[:rT])))
        
        results = {}
        
        results['A3IPW_mean'] = theta_a3ipw
        results['A3IPW_var'] = (1/np.mean(1/np.sqrt(self.est_var_list[:rT])))**2

        print('A3IPW', theta_a3ipw)
        
        #mixed a3ipw estimator
        #second stage estimation
        temp_term = self.A[gT:rT]*(self.Y[gT:rT] - mean_y_array[gT:rT])/pi_behavior_matrix[gT:rT] + mean_y_array[gT:rT]
        theta_temp = np.sum(self.pi_evaluation_matrix[gT:rT]*temp_term, axis=1)
                        
        theta_ma3ipw = np.sum(theta_temp/np.sqrt(self.est_var_list[gT:rT]))/(np.sum(1/np.sqrt(self.est_var_list[gT:rT])))
        
        results['MA3IPW_mean'] = theta_ma3ipw
        results['MA3IPW_var'] = (1/np.mean(1/np.sqrt(self.est_var_list[gT:rT])))**2
    
        #dm estimator
        theta_dm = np.mean(np.sum(self.pi_evaluation_matrix[:rT]* mean_y_array_temp[rT-1, :rT], axis=1))
        
        results['AdaDM_mean'] = theta_dm
        
        #adaptive variance
        adaptive_var_adadm_array =  np.sum(self.A*self.pi_evaluation_matrix*(self.Y - mean_y_array)/pi_behavior_matrix, axis=1)
        adaptive_var_adadm_array += np.sum(self.pi_evaluation_matrix*mean_y_array - theta_dm, axis=1)
        adaptive_var_adadm_array = adaptive_var_adadm_array**2
        adaptive_var_adadm= np.mean(adaptive_var_adadm_array)
        
        results['AdaDM_var'] = adaptive_var_adadm
        
        #adaipw estimatro
        temp_term_adaipw = self.A*self.Y/pi_behavior_matrix
        theta_adaipw = np.mean(np.sum(self.pi_evaluation_matrix[:rT]*temp_term_adaipw[:rT], axis=1))
        
        results['AdaIPW_mean'] = theta_adaipw
        
        #adaptive variance
        adaptive_var_ipw_array =  np.mean(np.sum(self.A[:rT]*self.pi_evaluation_matrix[:rT]*self.Y[:rT]/pi_behavior_matrix[:rT], axis=1)**2)
        adaptive_var_ipw_array += theta_adaipw**2
                
        results['AdaIPW_var'] = adaptive_var_ipw_array
        
        #a2ipw estimator
        temp_term_a2ipw = self.A*(self.Y - mean_y_array)/pi_behavior_matrix + mean_y_array
        theta_a2ipw = np.mean(np.sum(self.pi_evaluation_matrix[:rT]*temp_term_a2ipw[:rT], axis=1))
        
        results['A2IPW_mean'] = theta_a2ipw
        
        #adaptive variance
        adaptive_var_a2ipw_array =  np.sum(self.A*self.pi_evaluation_matrix*(self.Y - mean_y_array)/pi_behavior_matrix, axis=1)
        adaptive_var_a2ipw_array += np.sum(self.pi_evaluation_matrix*mean_y_array - theta_dm, axis=1)
        adaptive_var_a2ipw_array = adaptive_var_a2ipw_array**2
        adaptive_var_a2ipw= np.mean(adaptive_var_a2ipw_array[:rT])
                
        results['A2IPW_var'] = adaptive_var_a2ipw
        
        #a3daipw estimator
        #first stage estimation of theta
        theta_a3daipw_list = np.zeros(rT)
    
        for sample_idx in range(1, rT):
            temp_term = self.A[:sample_idx]*(self.Y[:sample_idx] - mean_y_array[:sample_idx])/pi_behavior_matrix[:sample_idx] + mean_y_array[:sample_idx]
            theta_temp = np.mean(np.sum(self.pi_evaluation_matrix[:sample_idx]*temp_term, axis=1))
            theta_a3daipw_list[sample_idx] = theta_temp
                            
        #first stage estimation of squared
        est_var_a3daipw_list = np.ones(rT)
        
        #\hat{var}_{t-1}
        squared_temp_array = mean_y2_array_temp
        
        for sample_idx in range(rT):
            #\hat{Var[y|x]}_{t-1}
            squared_temp = squared_temp_array[sample_idx, rT:]
            squared_temp[ squared_temp <= 0.1] = 0.1
            
            pi_temp = self.pi_behavior_array[sample_idx, rT:]
            
            temp_term1 = np.sum(self.pi_evaluation_matrix[rT:]**2* squared_temp/pi_temp, axis=1)
            temp_term2 = theta_a3daipw_list[sample_idx]**2

            est_var_a3daipw_list[sample_idx] = np.mean(temp_term1 + temp_term2)
        
        #second stage estimation        
        temp_term = self.A[:rT]*self.Y[:rT]/pi_behavior_matrix[:rT]
        theta_temp = np.sum(self.pi_evaluation_matrix[:rT]*temp_term, axis=1)
                        
        theta_a3daipw = np.sum(theta_temp/np.sqrt(est_var_a3daipw_list[:rT]))/(np.sum(1/np.sqrt(est_var_a3daipw_list[:rT])))
        
        results['Ada3IPW_mean'] = theta_a3daipw
        results['Ada3IPW_var'] = (1/np.mean(1/np.sqrt(est_var_a3daipw_list[:rT])))**2
        
        print('AdaDM', theta_dm)
        print('AdaIPW',  np.mean(theta_adaipw))
        print('A2IPW',  np.mean(theta_a2ipw))
        print('Ada3IPW',  np.mean(theta_a3daipw))
        
        return results
   
    #estimate related parameters
    def construct_function_list(self, method):
        #estimators using samples at each period
        function_y_array = np.zeros((self.sample_size, self.sample_size, self.num_actions))
        function_y2_array = np.zeros((self.sample_size, self.sample_size, self.num_actions))
        
        #for each sampe...
        for sample_idx in range(1, self.sample_size):
            #for each action...
            for action in self.action_sets:
                #find index of samples chose the action
                sample_with_action_a = np.where(self.A[:sample_idx, action]==1)[0]
                
                #set of samples
                Y_temp = self.Y[sample_with_action_a, action]
                X_temp = self.X[sample_with_action_a]
                
                #if we have at least two samples
                if len(sample_with_action_a) > 1:
                    
                    if method == 'NW_regression':
                        #model_y = KernelReg(Y_temp, X_temp, var_type='c'*self.dim, reg_type='lc')
                        #model_y2 = KernelReg(Y_temp**2, X_temp, var_type='c'*self.dim, reg_type='lc')
                        
                        model_y = KernelRegression(gamma=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])
                        model_y2 = KernelRegression(gamma=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

                    elif method == 'knn_regression':
                        model_y = KNeighborsRegressor(n_neighbors=np.int(np.sqrt(len(sample_with_action_a))))
                        model_y2 = KNeighborsRegressor(n_neighbors=np.int(np.sqrt(len(sample_with_action_a))))
                        
                    #train models
                    model_y.fit(X_temp, Y_temp)
                    model_y2.fit(X_temp, Y_temp**2)
                    
                    #estimate parameters
                    mean_y = model_y.predict(self.X)
                    mean_y2 = model_y2.predict(self.X)

                    function_y_array[sample_idx, :, action] = mean_y
                    function_y2_array[sample_idx, :, action] = mean_y2
            
            percent = sample_idx/self.sample_size
            if percent*100%10 == 0:
                print('Progress: ', '{:.0%}'.format(percent))
                
        self.function_y_dict[method] = function_y_array
        self.function_y2_dict[method] = function_y2_array


def main(arguments):
    args = process_args(arguments)

    sample_size = args.sample_size
    trial = args.num_trials

    tau = args.tau

    policy_type = args.policy_type


    data_name = args.dataset
    ratio = args.ratio
    gamma = args.gamma
    para_r = args.para_r
    para_r = ratio
    
    method = args.method

    Errors = np.zeros((6, trial))
    Vars = np.zeros((6, trial))

    print(int(para_r*sample_size))

    for trl in range(trial):
        covariates, true_outcome_matrix, action_sets = data_generation(data_name, sample_size)

        pi_behavior_array, pi_evaluation_matrix, chosen_action_matrix, observed_outcome_matrix  = behavior_and_evaluation_policy(covariates, true_outcome_matrix, action_sets,  policy_type=policy_type, tau=tau)

        theta_true = true_value(true_outcome_matrix, pi_evaluation_matrix)
        print('theta', theta_true)

        opee = OPE_Estimators(covariates, chosen_action_matrix, observed_outcome_matrix, action_sets, pi_behavior_array, pi_evaluation_matrix)
            
        results = opee.est_a3ipw(para_r=para_r, ratio=ratio, gamma=gamma, method=method)

        Errors[0, trl] = results['A3IPW_mean'] - theta_true
        Errors[1, trl] = results['MA3IPW_mean'] - theta_true
        Errors[2, trl] = results['AdaDM_mean'] - theta_true
        Errors[3, trl] = results['AdaIPW_mean'] - theta_true
        Errors[4, trl] = results['A2IPW_mean'] - theta_true
        Errors[5, trl] = results['Ada3IPW_mean'] - theta_true

        Vars[0, trl] = results['A3IPW_var']
        Vars[1, trl] = results['MA3IPW_var']
        Vars[2, trl] = results['AdaDM_var']
        Vars[3, trl] = results['AdaIPW_var']
        Vars[4, trl] = results['A2IPW_var']
        Vars[5, trl] = results['Ada3IPW_var']

        np.savetxt('results/errors_dataset_%s_policy_type_%s_method_%s_samplesize_%s_ratio_%3f_gamma_%3f.csv'%(data_name, policy_type, method, sample_size, ratio, gamma), Errors)
        np.savetxt('results/vars_dataset_%s_policy_type_%s_method_%s_samplesize_%s_ratio_%3f_gamma_%3f.csv'%(data_name, policy_type, method, sample_size, ratio, gamma), Vars)

if __name__ == '__main__':
    main(sys.argv[1:])
