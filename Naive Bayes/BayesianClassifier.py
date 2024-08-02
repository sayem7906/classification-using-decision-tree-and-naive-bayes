import numpy as np 
import pandas as pd     
import matplotlib.pyplot as plt 
import math


class  NaiveBayes:
    def __init__(self):
        pass

    def fit(self, X, y,attr):

        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = len(y)
        self.num_feats = len(X)
        self.likelihoods = {}
        self.class_priors = {}
        self.outcome_count = {}
        
        print("--------training--------")

        for feature in self.features:
            self.likelihoods[feature] = {}
            
            for feat_val in np.unique(self.X_train[feature]):

                for outcome in np.unique(self.y_train):
                    #print(outcome)
                    if(attr[feature]=='categorical'):
                        self.likelihoods[feature].update({str(feat_val)+'_'+str(outcome):0})
                    else:
                        self.likelihoods[feature].update({str(outcome)+'_mean':0})
                        self.likelihoods[feature].update({str(outcome)+'_variance':0})
                    self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods(attr)

    def _calc_class_prior(self):
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size
            self.outcome_count[outcome] = outcome_count

    def _calc_likelihoods(self,attr):
        for feature in self.features:

            for outcome in np.unique(self.y_train):
                index_outcome = self.y_train.where(self.y_train==outcome).dropna().index
                if(attr[feature] == 'categorical'):
                    outcome_count = sum(self.y_train == outcome)
                    feat_count = self.X_train[feature][index_outcome].value_counts().to_dict()
    
                    for feat_val in feat_count:
                        self.likelihoods[feature][str(feat_val)+'_'+str(outcome)] = feat_count[feat_val]/outcome_count
                else:
                    self.likelihoods[feature][str(outcome)+'_mean'] = self.X_train[feature][index_outcome].mean()
                    self.likelihoods[feature][str(outcome)+'_variance'] = self.X_train[feature][index_outcome].var()
                    


    def predict(self, X,attr):

        results = []
        X = np.array(X)
        #print(X)

        for query in X:
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1

                for feat, feat_val in zip(self.features, query):
                    if(attr[feat]=='categorical'):
                        x_y = str(feat_val)+'_'+str(outcome)
                        if(x_y in self.likelihoods[feat].keys()):
                            likelihood *= self.likelihoods[feat][x_y]
                        else:
                            likelihood *= 1/(self.outcome_count[outcome] + len(self.features))
                    else:
                        mean = self.likelihoods[feat][str(outcome)+'_mean']
                        var = self.likelihoods[feat][str(outcome)+'_variance']
                        if(var):
                            likelihood *= (1/math.sqrt(2*math.pi*var)) * np.exp(-(feat_val - mean)**2 / (2*var))
                        else:
                            likelihood = 0

                posterior = (likelihood * prior)
                probs_outcome[outcome] = posterior
                
            result = max(probs_outcome, key = lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)