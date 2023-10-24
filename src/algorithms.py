import copy
import os
import sys
from gurobipy import *
import numpy as np
from gurobipy import GRB
import gurobipy as gp
import pandas as pd
from argparser import *
from scipy.stats import norm
np.random.seed(10)
import random
random.seed(10)

class MDP():
    def __init__(self, initial, states, actions, rewards, transitions, gamma):
        """
        :param initial: initial distribution
        :param states: no of states
        :param actions: no of actions
        :param rewards: reward matrix SxAxS
        :param transitions: transition matrix SxAxS
        :param gamma: discount factor
        """
        self.ns = states
        self.na = actions
        self.p0 = initial
        self.rewards = rewards
        self.transitions = transitions
        self.gamma = gamma


class weighted_rmdp():
    def __init__(self,name, mdp, iters, p_bar, type,delta, p_samples, test_samples,true_model, original_data=None,seed=None, nsa=None):
        """
        :param name: name of experiment
        :param mdp: nominal mdp
        :param iters: number of iterations for value iteration
        :param p_bar: mean transition probability
        :param type: l1 or linf
        :param delta: confidence level
        :param p_samples: N transition model samples NxSxAxS
        :param test_samples: test transition model samples MxSxAxS
        :param true_model: true transition model SxAxS
        """
        self.name  = "new"+name + "-" + type + "-" + str(delta) + "-" + str(seed)

        if os.path.exists("logs/") is False:
            os.mkdir("logs/")
        self.dir = "logs/" + self.name
        if os.path.exists(self.dir) is False:
            os.mkdir(self.dir)
        self.w = np.ones(mdp.ns)
        self.nsa = nsa
        self.ns = mdp.ns
        self.mdp = mdp
        self.na = mdp.na
        self.gamma = mdp.gamma
        self.original_data = original_data
        self.value = np.zeros(self.ns)
        self.test_samples = test_samples
        self.iters=iters
        self.pbar = np.mean(p_samples,0)
        self.delta = delta
        self.alpha = delta/(self.ns)
        self.p_samples = p_samples
        self.p_bar = p_bar
        self.true_model = true_model
        self.mean_transition = self.compute_mean(p_samples)
        self.covariance = self.compute_covariance(p_samples)



    def compute_mean(self, p_samples):
        """
        Computes the mean transition probability matrix
        :param p_samples: transition models NxSxAxS
        :return: mean transition model SxAxS
        """
        return np.mean(p_samples,0)


    def compute_covariance(self,p_samples):
        """
        computes the covariance matrix from the N transition models
        :param p_samples: N transition models NxSxAxS
        :return: covariance matrix |SxA|x|SxA|
        """
        covariances = []
        shape = p_samples.shape
        for idx in range(shape[1]):
            covs = []
            for jdx in range(shape[2]):
                cov_ = np.cov(p_samples[:,idx,jdx,:].transpose())
                covs.append(cov_)
            covariances.append(covs)
        return np.array(covariances)

    def optimize_shape(self,type,v=None):
        """
        Optimizes the shape of BCR ambiguity set
        :param type: l1 or linf
        :param v: robust value function computed for unweighted BCR RMDP |S|
        :return: size (SxA) and optimal weights (SxAxS) of weighted BCR ambiguity set
        """
        if v is None:
            v, _ = self.compute_value(self.pbar)
        wts = (1.0/np.sqrt(self.ns))*np.ones((self.ns,self.na,self.ns))
        sizes = self.compute_size(self.p_samples,wts,type)
        z = self.mdp.rewards + self.mdp.gamma * v.reshape(1,1,self.ns)
        lam = (np.max(z,axis=-1) + np.min(z,axis=-1))/2

        weights = self.compute_weights(z,lam,type)

        self.w = weights
        new_size = self.compute_size(self.p_samples,weights,type)
        self.size = new_size
        return new_size, weights

    def compute_weights(self, z,lam,type):
        """
        Compute weights for given realization of z, lam
        :param z: r_{s,a} + \gamma v
        :param lam: lambda in http://proceedings.mlr.press/v130/behzadian21a/behzadian21a.pdf
        :param type: l1 or linf
        :return:
        """
        lam = lam.reshape(self.ns,self.na,1)
        if type == "l1":
            num = np.power(np.abs(z-lam),1.0/3)
            den = np.sqrt(np.sum(np.power(np.abs(z-lam),2.0/3),axis=-1))
            den = den.reshape(self.ns,self.na,1)
            w = num/den
        else:
            num = np.abs(z - lam)
            den = np.sqrt(np.sum(np.power(np.abs(z - lam), 2.0),axis=-1))
            den = den.reshape(self.ns,self.na,1)
            w = num / den
        return w

    def compute_value(self,transitions):
        """
        Computes value function for the given transition model
        :param transitions: SxAxS transition model
        :return: value function |S| and policy |SxA|
        """
        m = Model()
        value = m.addVars(self.ns,vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="value")
        m.setObjective(gp.quicksum(self.mdp.p0[r] * value[r] for r in np.arange(self.ns)), GRB.MINIMIZE)
        for idx in np.arange(self.ns):
            for jdx in np.arange(self.na):
                m.addConstr(value[idx] >= gp.quicksum(self.mdp.rewards[idx,jdx,kdx] + self.gamma * value[kdx] * transitions[idx,jdx,kdx] for kdx in np.arange(self.ns)), "c1"+str(idx)+"-"+str(jdx))

        m.optimize()
        print('Maximized value:', m.objVal)
        val = np.array([v.X for k,v in value.items() ])

        policy = np.zeros((self.ns,self.na))
        qsa_ = self.mdp.rewards + self.mdp.gamma * val.reshape((1, 1, -1))
        qsa_temp = qsa_.reshape((-1, self.ns))
        qsa = np.sum(transitions.reshape(-1, self.ns) * qsa_temp, -1).reshape((self.ns, self.na))

        actions = np.argmax(qsa, axis=-1)
        policy[np.arange(self.ns), actions.flatten()] = 1.0



        return val, policy

    def compute_true_value(self):
        """
        Computes value on the mean model
        :return: value function |S| and policy |SxA|
        """

        value, policy = self.compute_value(self.mdp.transitions)
        policy_name = self.dir + "/" + "_true_policy.npy"
        np.save(policy_name, np.array(policy))
        return value, policy

    def compute_returns(self,pi,rew,transitions):
        """
        Computes
        :param pi: policy
        :param rew: reward matrix SxAxS
        :param transitions:  transition matrix SxAxS
        :return: returns |S|
        """
        I = np.eye(self.ns)
        pi_p_temp = pi.reshape((self.ns, self.na, 1)) * transitions
        pi_p = np.sum(pi_p_temp, axis=1)
        pi_p_inverse = np.linalg.inv(I - self.mdp.gamma * pi_p)
        r = np.sum(np.sum(pi_p_temp * rew, -1), -1)
        ret = np.matmul(pi_p_inverse, r)
        return ret

    def compute_value_true_model(self,policy,name):
        """
        Computes returns on true model
        :param policy: deterministic policy |SxA|
        :param name: name of the method
        :return: returns |S|
        """
        rets_name = self.dir + "/" +  name + "_rets.npy"
        rets = self.compute_returns(policy,self.mdp.rewards,self.true_model)
        np.save(rets_name, np.array(rets))
        return rets

    def compute_value_mean_model(self,policy,name):
        """
        Computes returns on mean model
        :param policy: policy |SxA|
        :param name: name of method
        :return: returns |S|
        """
        rets_name = self.dir + "/" +  name + "_rets.npy"
        rets = self.compute_returns(policy, self.mdp.rewards, self.pbar)
        np.save(rets_name, np.array(rets))
        return rets

    def compute_mean_policy_value(self):
        """
        Computes policy on mean model
        :return: value function of the policy |S| and policy |SxA|
        """

        value, policy = self.compute_value(self.pbar)
        policy_name = self.dir + "/" + "_mean_policy.npy"
        value_name = self.dir + "/" + "_mean_value.npy"
        np.save(policy_name, np.array(policy))
        np.save(value_name,np.array(value))
        return value, policy

    def compute_optimized_bcr(self,type,v):
        """
        Computes value function for optimized weighted BCR ambiguity set
        :param type: l1 or linf
        :param v: robust value function for unweighted BCR ambiguity set
        :return: value function |S| and policy |SxA|
        """
        size, weights = self.optimize_shape(type,v)
        if type =="l1":
            value, policy = self.robust_value_iteration(size,weights)
        else:
            value, policy = self.linf_robust_value_iteration(size, weights)
        policy_name = self.dir + "/" + type + "_optimized_robust_policy.npy"
        np.save(policy_name, np.array(policy))
        value_name = self.dir + "/" + type + "_optimized_robust_value.npy"
        np.save(value_name,np.array(value))
        return value, policy




    def compute_naive_bcr(self,type):
        """
        Computes the value function for the unweighted BCR RMDP
        :param type: l1 or linf
        :return: value function |S| and policy |SxA|
        """

        weights = np.ones((self.ns,self.na,self.ns))
        size = self.compute_size(self.p_samples, weights,type)

        if type =="l1":
            value, policy = self.robust_value_iteration(size,weights)
        else:
            value, policy = self.linf_robust_value_iteration(size, weights)
        policy_name = self.dir + "/" + type + "_naive_robust_policy.npy"
        np.save(policy_name, np.array(policy))
        value_name = self.dir + "/" + type + "_naive_robust_value.npy"
        np.save(value_name, np.array(value))
        return value, policy


    def compute_size_hoeffding(self):
        temp = (self.ns * self.na * (2 ** self.ns)) / self.delta
        log_temp = np.log(temp)
        sizes = np.zeros((self.ns,self.na))
        for idx in range(self.ns):
            for jdx in range(self.na):

                nsa = self.nsa
                sizes[idx,jdx]= np.sqrt((2.0/(max(1.0,nsa[idx,jdx])))*log_temp)
        return sizes

    def compute_size_optimized(self):
        temp = (self.ns * self.na * self.ns) / self.delta
        log_temp = np.log(temp)
        sizes = np.zeros((self.ns, self.na))
        for idx in range(self.ns):
            for jdx in range(self.na):
                nsa = self.nsa
                sizes[idx, jdx] = np.sqrt((2.0 / (max(nsa[idx,jdx],1))) * log_temp)
        return sizes

    def compute_hoeffding(self,type):
        """
        Computes the value function for the unweighted BCR RMDP
        :param type: optimized naive
        :return: value function |S| and policy |SxA|
        """

        weights = np.ones((self.ns,self.na,self.ns))
        if type=="optimized":
            size = self.compute_size_optimized()
        else:
            size = self.compute_size_hoeffding()

        #l1 norm ball robust value iteration
        value, policy = self.robust_value_iteration(size,weights)

        policy_name = self.dir + "/" + type + "_hoeffding_policy.npy"
        np.save(policy_name, np.array(policy))
        value_name = self.dir + "/" + type + "_hoeffding_value.npy"
        np.save(value_name, np.array(value))
        return value, policy





    def robust_value_iteration(self,size,weights,eps=1e-3):
        """
        Robust Value iteration algorithm for computing value function and policy for Robust MDP with l1-weighted
        ambiguity sets
        :param size: size of the ambiguity sets S x A
        :param weights: weights of the ambiguity sets SxAxS
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        oldval = None
        value = np.random.uniform(0,1,(self.ns))
        # value = np.zeros(0,1,(self.ns))

        iter=0
        policy=None
        while(oldval is None or np.max(np.abs(oldval-value))>eps):
            policy = np.zeros((self.ns,self.na))
            p= np.zeros((self.ns, self.na, self.ns))
            for idx in range(self.ns):
                for jdx in range(self.na):

                    m = Model()

                    psa = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=0,
                                    ub=GRB.INFINITY, name="transition")
                    psa_abs = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, name="abs_transition")

                    psa_temp = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=0,
                                        ub=GRB.INFINITY, name="temp_transition")

                    m.setObjective(gp.quicksum(psa[ kdx] * (self.mdp.rewards[idx,jdx, kdx] + self.gamma *
                                                                     value[kdx]) for kdx in np.arange(self.ns)), GRB.MINIMIZE)
                    for kdx in np.arange(self.ns):
                        m.addConstr(psa_abs[ kdx] ==  psa[kdx]-self.pbar[idx,jdx,kdx],
                                    "c1" + str(idx) + str(jdx) + str(kdx))
                        m.addConstr(psa_temp[ kdx] == gp.abs_(psa_abs[kdx]),
                                    "c2" + str(idx) + str(jdx) + str(kdx))

                    m.addConstr(gp.quicksum(weights[idx,jdx,kdx]* psa_temp[kdx] for kdx in range(self.ns)) <= size[idx,jdx],"c3" + str(idx) + str(jdx))
                    # m.addConstr(gp.quicksum(self.w[idx,jdx,kdx]* psa_abs[idx,jdx,kdx]  for kdx in range(self.ns)) >= -1.0*self.size[idx,jdx],"c3" + str(idx) + str(jdx))

                    m.addConstr(gp.quicksum(psa[kdx] for kdx in range(self.ns))==1.0,"c4" + str(idx) + str(jdx))
                    for kdx in range(self.ns):
                        m.addConstr(psa[kdx]>=0,"c5" + str(idx) + str(jdx)+ str(kdx))
                    m.setParam('OutputFlag', 0)
                    m.optimize()
                    p[idx,jdx,:] = np.array([v.X for k,v in psa.items() ])

            qsa_ = self.mdp.rewards + self.mdp.gamma * value.reshape((1,1,-1))
            qsa_temp = qsa_.reshape((-1,self.ns))
            qsa = np.sum(p.reshape(-1,self.ns)*qsa_temp,-1).reshape((self.ns,self.na))
            oldval = value
            value = np.max(qsa,axis=-1)
            actions = np.argmax(qsa,axis=-1)
            policy[np.arange(self.ns),actions.flatten()]=1.0


            print("robust value iter",iter)
            print("old val - value", np.max(np.abs(oldval-value)))
            iter+=1
            if iter > 2000:
                break

        ret = np.sum(self.mdp.p0.flatten() * value.flatten())
        print("var robust returns ", ret)

        return value, policy



    def linf_robust_value_iteration(self,size,weights,eps=1e-3):
        """
        Robust value iteration for Robust MDPs with $linf$-weighted ambiguity sets
        :param size: size of ambiguity sets |SxA|
        :param weights: weights of ambiguity sets |SxAxS|
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        oldval = None
        value = np.random.uniform(0,1,(self.ns))
        iter=0
        policy=None
        while(oldval is None or np.max(np.abs(oldval-value))>eps):
            policy = np.zeros((self.ns,self.na))
            p= np.zeros((self.ns, self.na, self.ns))
            for idx in range(self.ns):
                for jdx in range(self.na):

                    m = Model()

                    psa = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                    ub=GRB.INFINITY, name="transition")
                    psa_abs = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                        ub=GRB.INFINITY, name="abs_transition")

                    psa_temp = m.addVars(self.ns, vtype=GRB.CONTINUOUS, lb=0,
                                        ub=GRB.INFINITY, name="temp_transition")

                    m.setObjective(gp.quicksum(psa[ kdx] * (self.mdp.rewards[idx,jdx, kdx] + self.gamma *
                                                                     value[kdx]) for kdx in np.arange(self.ns)), GRB.MINIMIZE)
                    for kdx in np.arange(self.ns):
                        m.addConstr(psa_abs[ kdx] ==  psa[kdx]-self.pbar[idx,jdx,kdx],
                                    "c1" + str(idx) + str(jdx) + str(kdx))
                        m.addConstr(psa_temp[ kdx] == gp.abs_(psa_abs[kdx]),
                                    "c2" + str(idx) + str(jdx) + str(kdx))


                        m.addConstr(weights[idx,jdx,kdx]* psa_temp[kdx] <= size[idx,jdx],"c3" + str(idx) + str(jdx))
                    # m.addConstr(gp.quicksum(self.w[idx,jdx,kdx]* psa_abs[idx,jdx,kdx]  for kdx in range(self.ns)) >= -1.0*self.size[idx,jdx],"c3" + str(idx) + str(jdx))

                    m.addConstr(gp.quicksum(psa[kdx] for kdx in range(self.ns))==1.0,"c4" + str(idx) + str(jdx))
                    for kdx in range(self.ns):
                        m.addConstr(psa[kdx]>=0,"c5" + str(idx) + str(jdx)+ str(kdx))
                    m.setParam('OutputFlag', 0)
                    m.optimize()
                    p[idx,jdx,:] = np.array([v.X for k,v in psa.items() ])


            qsa_ = self.mdp.rewards + self.mdp.gamma * value.reshape((1,1,-1))
            qsa_temp = qsa_.reshape((-1,self.ns))
            qsa = np.sum(p.reshape(-1,self.ns)*qsa_temp,-1).reshape((self.ns,self.na))
            oldval = value
            value = np.max(qsa,axis=-1)
            actions = np.argmax(qsa,axis=-1)
            policy[np.arange(self.ns),actions.flatten()]=1.0


            print("robust value iter",iter)
            print("old val - value", np.max(np.abs(oldval-value)))
            iter+=1
            if iter > 2000:
                break

        ret = np.sum(self.mdp.p0.flatten() * value.flatten())
        print("var robust returns ", ret)

        return value, policy

    def extract_policy(self, values, p, rew):
        """
        Extract policy from value function
        :param values: value function |S|
        :param p: transition probability model |SxAxS|
        :param rew: reward |SxAxS|
        :return: policy |SxA|
        """
        policy = np.zeros((self.ns,self.na))

        temp = rew + self.gamma * values.reshape(1,1,self.ns)
        temp = temp.reshape(-1,self.ns)
        res = np.sum(p.reshape(-1,self.ns)*temp,axis=-1)
        val = res.reshape((self.ns,self.na))
        for idx in range(self.ns):
            a = np.argmax(val,-1)
            policy[idx,a]=1.0
        return policy

    def evaluate_all(self,pi ,rew,type, test=True):
        """
        Computes returns on train/test transition models
        :param pi: policy |SxA|
        :param rew: reward matrix |SxAxS|
        :param type: l1 or linf
        :param test: True if to be evaluated on test transition models
        :return: returns |NxS|
        """
        if test == True:
            psamples_test = self.test_samples
        else:
            psamples_test = self.p_samples
        num = psamples_test.shape[0]
        returns = []
        I = np.eye(self.ns)
        for idx in range(num):
            pi_p_temp = pi.reshape((self.ns,self.na,1)) * psamples_test[idx,:,:,:]
            pi_p = np.sum(pi_p_temp,axis=1)
            pi_p_inverse = np.linalg.inv(I - self.mdp.gamma *pi_p)
            r = np.sum(np.sum(pi_p_temp* rew,-1),-1)
            ret = np.matmul(pi_p_inverse,r)
            returns.append(np.dot(self.mdp.p0, ret))
        if test is True:
            fname = self.dir + "/" + type + "_test_rets.npy"
        else:
            fname = self.dir + "/" + type +"_train_rets.npy"
        np.save(fname,np.array(returns))
        return returns

    def evaluate_true_model(self,pi,true_model):
        """
        Evaluate returns on true model
        :param pi: policy |SxA|
        :param true_model: true transition model |SxAxS|
        :return: returns |S|
        """

        I = np.eye(self.ns)
        rew = self.mdp.rewards

        pi_p_temp = pi.reshape((self.ns, self.na, 1)) * true_model
        pi_p = np.sum(pi_p_temp, axis=1)
        pi_p_inverse = np.linalg.inv(I - self.mdp.gamma * pi_p)
        r = np.sum(np.sum(pi_p_temp * rew, -1), -1)
        ret = np.matmul(pi_p_inverse, r)
        returns = np.array([np.dot(self.mdp.p0, ret)])
        print("true_returns",returns)
        fname = self.dir + "/" + "true_returns.npy"
        np.save(fname, np.array(returns))
        return returns

    def compute_asymptotic_var(self,s,a,v):
        """
        Computes asymptotic value at risk for a given state s and action a and value function v
        :param s: state (integer)
        :param a: action (integer)
        :param v: value function |S|
        :return: Asymptotic VaR for state s and action a computed with value function v
        """
        mean = self.mean_transition[s,a,:]
        cov = self.covariance[s,a,:,:]
        v = v.reshape(-1,1)
        temp = np.matmul(v.transpose(),cov).reshape(1,-1)
        temp1 = np.matmul(temp,v)
        temp2 = np.sqrt(temp1)
        VaR = np.dot(mean,v) + norm.ppf(self.alpha)* temp2

        return VaR

    def asymptotic_var_value_iteration(self,eps=1e-4):
        """
        Asymptotic VaR value iteration
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        nsamples = self.p_samples.shape[0]
        v = np.random.uniform(0, 1, self.ns)

        vold = None
        iter = 0
        while True:
            policy = np.zeros((self.ns, self.na))
            for s in range(self.ns):
                res = []
                for a in range(self.na):
                    values = []

                    for sid in range(nsamples):
                        val = np.dot(self.p_samples[sid, s, a, :].flatten(),
                                     self.mdp.rewards[s, a, :] + self.mdp.gamma * v.flatten())
                        values.append(val)
                    buffer_r = self.mdp.rewards[s, a, :] + self.mdp.gamma * v.flatten()
                    var = self.compute_asymptotic_var(s,a,buffer_r)
                    res.append(var)
                v[s] = np.max(res)
                policy[s, np.argmax(res)] = 1.0
            if vold is not None:
                if np.max(np.abs(vold - v)) < eps:
                    break
            iter += 1
            if iter > 2000:
                break
            print("asymptotic var robust iter", iter)
            vold = copy.deepcopy(v)
        ret = np.sum(self.mdp.p0.flatten() * v.flatten())
        print("asymptotic var robust returns*** ", ret)
        policy_name = self.dir + "/" + "asymptoticvar_policy.npy"
        val_name = self.dir + "/" + "asymptoticvar_value.npy"
        np.save(val_name, np.array(v))
        np.save(policy_name, np.array(policy))

        return v, policy


    def var_robust_value_iteration(self, eps=1e-4):
        """
        Implementation of generalized VaR value iteration
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        nsamples = self.p_samples.shape[0]
        v = np.random.uniform(0,1,self.ns)
        # v = np.zeros((self.ns))


        vold = None
        iter=0
        while True:
            policy = np.zeros((self.ns,self.na))
            for s in range(self.ns):
                res=[]
                for a in range(self.na):
                    values = []

                    for sid in range(nsamples):
                        val = np.dot(self.p_samples[sid,s,a,:].flatten(),self.mdp.rewards[s,a,:] + self.mdp.gamma *v.flatten())
                        values.append(val)
                    var = np.percentile(values,self.alpha)
                    res.append(var)
                v[s] = np.max(res)
                policy[s,np.argmax(res)]=1.0
            if vold is not None:
                if np.max(np.abs(vold - v)) < eps:
                    break
            iter +=1
            if iter > 2000:
                break
            print("var robust iter",iter)
            vold = copy.deepcopy(v)
        ret = np.sum(self.mdp.p0.flatten()* v.flatten())
        print("var robust returns*** ",ret)
        policy_name = self.dir + "/" + "var_policy.npy"
        val_name = self.dir + "/" + "var_value.npy"
        np.save(val_name, np.array(v))
        np.save(policy_name, np.array(policy))

        return v, policy

    def compute_cvar(self, rets, alpha):
        rets = np.array(rets)
        var_alpha = np.percentile(rets, alpha)
        cvar = rets[rets <= var_alpha].mean()
        return cvar






    def cvar_robust_value_iteration(self, eps=1e-3):
        """
        Implementation of generalized VaR value iteration
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        nsamples = self.p_samples.shape[0]
        v = np.random.uniform(0, 1, self.ns)
        # v = np.zeros((self.ns))

        vold = None
        iter = 0
        while True:
            policy = np.zeros((self.ns, self.na))
            for s in range(self.ns):
                res = []
                for a in range(self.na):
                    values = []

                    for sid in range(nsamples):
                        val = np.dot(self.p_samples[sid, s, a, :].flatten(),
                                     self.mdp.rewards[s, a, :] + self.mdp.gamma * v.flatten())
                        values.append(val)
                    cvar = self.compute_cvar(values,self.alpha)
                    res.append(cvar)
                v[s] = np.max(res)
                policy[s, np.argmax(res)] = 1.0
            if vold is not None:
                if np.max(np.abs(vold - v)) < eps:
                    break
            iter += 1
            if iter> 2000:
                break
            print("cvar robust iter", iter)
            vold = copy.deepcopy(v)
        ret = np.sum(self.mdp.p0.flatten() * v.flatten())
        print("cvar robust returns*** ", ret)
        policy_name = self.dir + "/" + "cvar_policy.npy"
        val_name = self.dir + "/" + "cvar_value.npy"
        np.save(val_name, np.array(v))
        np.save(policy_name, np.array(policy))

        return v, policy

    def soft_robust_value_iteration(self, eps=1e-4):
        """
        Implementation of generalized VaR value iteration
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        nsamples = self.p_samples.shape[0]
        v = np.random.uniform(0, 1, self.ns)
        # v = np.zeros((self.ns))

        vold = None
        iter = 0
        while True:
            policy = np.zeros((self.ns, self.na))
            for s in range(self.ns):
                res = []
                for a in range(self.na):
                    values = []

                    for sid in range(nsamples):
                        val = np.dot(self.p_samples[sid, s, a, :].flatten(),
                                     self.mdp.rewards[s, a, :] + self.mdp.gamma * v.flatten())
                        values.append(val)
                    mean_value = np.mean(values)
                    res.append(mean_value)
                v[s] = np.max(res)
                policy[s, np.argmax(res)] = 1.0
            if vold is not None:
                if np.max(np.abs(vold - v)) < eps:
                    break
            iter += 1
            if iter > 2000:
                break
            print("cvar robust iter", iter)
            vold = copy.deepcopy(v)
        ret = np.sum(self.mdp.p0.flatten() * v.flatten())
        print("cvar robust returns*** ", ret)
        policy_name = self.dir + "/" + "softrobust_policy.npy"
        val_name = self.dir + "/" + "softrobust_value.npy"
        np.save(val_name, np.array(v))
        np.save(policy_name, np.array(policy))

        return v, policy


    def worst_robust_value_iteration(self, eps=1e-4):
        """
        Implementation of generalized VaR value iteration
        :param eps: value iteration error
        :return: value function |S| and policy |SxA|
        """
        nsamples = self.p_samples.shape[0]
        v = np.random.uniform(0, 1, self.ns)
        # v = np.zeros((self.ns))

        vold = None
        iter = 0
        while True:
            policy = np.zeros((self.ns, self.na))
            for s in range(self.ns):
                res = []
                for a in range(self.na):
                    values = []

                    for sid in range(nsamples):
                        val = np.dot(self.p_samples[sid, s, a, :].flatten(),
                                     self.mdp.rewards[s, a, :] + self.mdp.gamma * v.flatten())
                        values.append(val)
                    minimum = np.min(values)
                    res.append(minimum)
                v[s] = np.max(res)
                policy[s, np.argmax(res)] = 1.0
            if vold is not None:
                if np.max(np.abs(vold - v)) < eps:
                    break
            iter += 1
            if iter > 2000:
                break
            print("cvar robust iter", iter)
            vold = copy.deepcopy(v)
        ret = np.sum(self.mdp.p0.flatten() * v.flatten())
        print("cvar robust returns*** ", ret)
        policy_name = self.dir + "/" + "worstrobust_policy.npy"
        val_name = self.dir + "/" + "worstrobust_value.npy"
        np.save(val_name, np.array(v))
        np.save(policy_name, np.array(policy))

        return v, policy

    def compute_size(self, p_samples, wts,type):
        """
        Computes size of the BCR ambiguity sets given transition probability samples, weights, and type of ambiguity sets
        :param p_samples: transition models |NxSxAxS|
        :param wts: weights of BCR ambiguity sets |SxAxS|
        :param type: l1 or linf
        :return: sizes of ambiguity sets |SxA|
        """
        sizes=[]
        if type=="l1":
            p_bar = np.mean(p_samples, axis=0)
            for idx in range(self.ns):
                for jdx in range(self.na):
                    ds = np.sum(wts[idx,jdx,].reshape(1,-1)* np.abs(p_bar[idx,jdx,:].reshape((1,-1)) - p_samples[:,idx,jdx,:]),axis=-1)
                    q = np.quantile(ds,1-self.alpha)
                    sizes.append(q)
            sizes = np.array(sizes).reshape((self.ns,self.na))

            return sizes
        else:
            p_bar = np.mean(p_samples, axis=0)
            for idx in range(self.ns):
                for jdx in range(self.na):
                    ds = np.max(wts[idx,jdx,].reshape(1,-1)* np.abs(p_bar[idx,jdx,:].reshape((1,-1)) - p_samples[:,idx,jdx,:]),axis=-1)
                    q = np.quantile(ds, 1-self.alpha)
                    sizes.append(q)
            sizes = np.array(sizes).reshape((self.ns, self.na))
            return sizes


def validate_transitionp(p, ns, na):
    """
    Validates if a given transition probability matrix is valid
    :param p: transition probability matrix |SxAxS|
    :param ns: number of states
    :param na: number of actions
    :return: True if the transition probability matrix is valid, False otherwise
    """
    if np.any(p<0) == True:
        return False
    for idx in range(ns):
        for jdx in range(na):
            if np.sum(p[idx,jdx,:])==1.0:
                continue
            else:
                return False

    return True

def load_nsa(path, ns, na):
    try:
        filepath = path + "/samples.csv"
        df = pd.read_csv(filepath)
    except:
        return None
    nsa = np.zeros((ns,na,ns))
    lent = len(df)
    for idx in range(lent):
        s = df['idstatefrom'][idx]
        a = df['idaction'][idx]
        ns = df['idstateto'][idx]
        nsa[s,a,ns]+=1
    print("nsa", nsa)
    return nsa

def load_initial(path):
    df = pd.read_csv(path + "/initial.csv")
    lent = len(df)
    init = np.zeros(lent)
    ids = df['idstate'].to_numpy()
    probs =  df['probability'].to_numpy()
    init[ids] = probs
    return init

def load_parameters(path):
    df = pd.read_csv(path + "/parameters.csv")
    gamma = df['value'][0]
    return gamma

def load_train_data(path,seed=0):
    df = pd.read_csv(path + "/training_" + str(seed) + ".csv")
    idstatefrom = df['idstatefrom']
    idaction = df['idaction']
    idstateto = df['idstateto']
    idoutcome = df['idoutcome']
    probability = df['probability']
    reward = df['reward']
    lent = len(df)
    ns = len(idstatefrom.unique())
    na = len(idaction.unique())
    no = len(idoutcome.unique())
    psamples = np.zeros((no,ns,na,ns))
    rewards = np.zeros((no,ns,na,ns))
    for idx in range(lent):
        id = idoutcome[idx]
        s = idstatefrom[idx]
        a = idaction[idx]
        s_ = idstateto[idx]
        psamples[id,s,a,s_]= probability[idx]
        rewards[id,s,a,s_] = reward[idx]
    nsa = load_nsa(path,ns,na)



    return psamples, rewards, nsa


def load_test_data(path):
    df = pd.read_csv(path + "/test.csv")
    idstatefrom = df['idstatefrom']
    idaction = df['idaction']
    idstateto = df['idstateto']
    idoutcome = df['idoutcome']
    probability = df['probability']
    reward = df['reward']
    lent = len(df)
    ns = len(idstatefrom.unique())
    na = len(idaction.unique())
    no = len(idoutcome.unique())
    psamples = np.zeros((no, ns, na, ns))
    rewards = np.zeros((no, ns, na, ns))
    for idx in range(lent):
        id = idoutcome[idx]
        s = idstatefrom[idx]
        a = idaction[idx]
        s_ = idstateto[idx]
        psamples[id, s, a, s_] = probability[idx]
        rewards[id, s, a, s_] = reward[idx]

    return psamples, rewards


def load_true(path):
    df = pd.read_csv(path + "/true.csv")
    idstatefrom = df['idstatefrom']
    idaction = df['idaction']
    idstateto = df['idstateto']
    probability = df['probability']
    reward = df['reward']
    lent = len(df)
    ns = len(idstatefrom.unique())
    na = len(idaction.unique())
    psample = np.zeros((ns, na, ns))
    rewards = np.zeros((ns, na, ns))
    for idx in range(lent):
        s = idstatefrom[idx]
        a = idaction[idx]
        s_ = idstateto[idx]
        psample[s, a, s_] = probability[idx]
        rewards[s, a, s_] = reward[idx]
    return psample, rewards




def load_all(env,delta,seed, nsa=None):

    name = env + "-" + str(delta)
    path = "Domains/" + env
    initial = load_initial(path)
    gamma = load_parameters(path)
    psamples, rewards, nsa = load_train_data(path,seed)
    if env == "inventory":
        s = rewards.shape[0]
        a = rewards.shape[1]
        nsa = np.ones((s,a))*4
    else:
        nsa = np.sum(nsa,2)



    psamples_test, rewards_test = load_test_data(path)
    true_model, reward = load_true(path)
    shape = psamples.shape
    pbar = np.mean(psamples,0)
    true = validate_transitionp(pbar,shape[1],shape[2])


    mdp = MDP(initial,psamples.shape[1], psamples.shape[2], reward,pbar, gamma)
    rmdp = weighted_rmdp(name,mdp, 1500, pbar, "l1", delta, psamples,psamples_test,true_model,seed=seed, nsa=nsa)


    all_policies=[]
    # v8, policy8 = rmdp.asymptotic_var_value_iteration()
    # all_policies.append(("asymptoticvar",policy8,v8))
    #
    # v1, policy1 = rmdp.var_robust_value_iteration()
    # all_policies.append(("var",policy1,v1))
    #
    # v9, policy9 = rmdp.cvar_robust_value_iteration()
    # all_policies.append(("cvar",policy9,v9))
    #
    # v12, policy12 = rmdp.soft_robust_value_iteration()
    # all_policies.append(("softrobust", policy12, v12))
    #
    # v13, policy13 = rmdp.worst_robust_value_iteration()
    # all_policies.append(("worstrobust", policy13, v13))
    #

    # v2, policy2 = rmdp.compute_naive_bcr("l1")
    # all_policies.append(("naive_bcr_l1", policy2, v2))
    # #
    # v3, policy3 = rmdp.compute_optimized_bcr("l1", v2)
    # all_policies.append(("weighted_bcr_l1", policy3, v3))

    # if env=="inventory":
    v11, policy11  = rmdp.compute_hoeffding("optimized")
    all_policies.append(("optimized_hoeffding", policy11, v11))

    v10, policy10 = rmdp.compute_hoeffding("naive")
    all_policies.append(("naive_hoeffding", policy10, v10))


    # v4, policy4 = rmdp.compute_naive_bcr("linf")
    # all_policies.append(("naive_bcr_linf",policy4,v4))
    #
    # v5, policy5 = rmdp.compute_optimized_bcr("linf",v4)
    # all_policies.append(("weighted_bcr_linf",policy5,v5))
    #

    #
    #
    # v6, policy6 = rmdp.compute_true_value()
    # all_policies.append(("true_model",policy6,v6))
    #
    # v7, policy7 = rmdp.compute_mean_policy_value()
    # all_policies.append(("mean_model",policy7,v7))
    #


    # policy11=None
    # policy10 = None

    # rmdp.evaluate_true_model(policy6,true_model)


    for res in all_policies:

        method = res[0]
        # if env!="inventory" and "hoeffding" in method:
        #     continue
        policy_ = res[1]
        value = res[2]
        train_rets = rmdp.evaluate_all(policy_, mdp.rewards, method, test=False)
        test_rets = rmdp.evaluate_all(policy_, mdp.rewards, method, test=True)
        print(str(method))
        print("train percentile returns",np.percentile(train_rets, delta))
        print("test percentile returns",np.percentile(test_rets, delta))
        rmdp.compute_value_true_model(policy_, method)
        rmdp.evaluate_true_model(policy_,true_model)
        # print("true value",)














    #


