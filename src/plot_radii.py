import os

import matplotlib.pyplot as plt
import numpy.random
import scipy
import numpy as np
numpy.random.seed(10)
import random
random.seed(10)
import gurobipy as gp

from scipy.stats import dirichlet
from scipy.stats import chi2
from scipy.stats import norm
plt.style.use('plot_style3.txt')

def compute_covariance(alpha):
    sum_a = np.sum(alpha)
    norm_a = alpha/sum_a
    arr=[]
    l = len(alpha)
    for idx in range(l):
        arr_temp = []
        for jdx in range(l):
            if idx==jdx:
                k = (norm_a[idx]-norm_a[idx]*norm_a[jdx])/(sum_a+1)
            else:
                k = (-norm_a[idx] * norm_a[jdx]) / (sum_a + 1)
            arr_temp.append(k)

        arr.append(arr_temp)
    return np.array(arr)

def compute_dist(z,M):
    M = np.linalg.pinv(M)
    temp1 = np.matmul(z,M)
    temp2 = np.matmul(temp1,z)
    r = np.sqrt(temp2).flatten()[0]
    return r

def validate_bcr(z,delta,ns,cov):
    r = compute_dist(z,cov)
    rad = np.sqrt(chi2.ppf(delta, ns, loc=0, scale=1))
    if np.abs(r)<= np.abs(rad):
        return True
    else:
        return False


def validate_var(z,delta,cov):
    r = compute_dist(z, cov)
    rad = norm.ppf(delta)
    if np.abs(r)<= np.abs(rad):
        return True
    else:
        return False

def plot_radii():
    e= 0.1 * np.arange(5)
    plt.figure(figsize=(7, 6.6))

    vals = chi2.ppf(e, df=150)

    dfs = np.array([1,2,3,4,5])*100
    vals1=[]
    plt.clf()
    plt.xlabel("confidence level 1-ε",fontsize=18)
    plt.ylabel("Ratio of asymptotic radii ξ",fontsize=18)
    plt.grid()
    # plt.grid(b=None)

    for df in dfs:
        v = chi2.ppf(e, df=df)
        z = norm.ppf(e, loc=0, scale=1)
        radi = np.abs(np.sqrt(v) / z)
        plt.annotate("S="+ str(df),xy=(e[-1],radi[-1]), fontsize=16)
        plt.plot(e,radi)
        plt.ylim(-1,95)
        plt.xlim(0,0.5)


    # plt.legend()
    plt.savefig("radii.pdf")

    # plot_returns(100)

def plot_returns(num_s=10):
    nums = np.arange(1,num_s,2).astype(int)
    rets_var = []
    rets_bcr = []
    rewards = np.random.rand(num_s)

    for num in nums:
        ns=num
        rew = rewards[:num]
        alpha = np.ones(num)
        variables = dirichlet.rvs(alpha, size=10000, random_state=1)
        mean = dirichlet.mean(alpha)

        points_bcr = []
        delta = 0.8
        cov = compute_covariance(alpha)
        points_bcr_ = []
        for pt in variables:
            p = pt - mean
            if (validate_var(p, delta, cov) == False and validate_bcr(p, delta, ns, cov) == True):
                points_bcr.append(pt)

            if (validate_bcr(p, delta, ns, cov) == True):
                points_bcr_.append(pt)

        points_var = []
        for pt in variables:
            p = pt - mean
            if validate_var(p, delta, cov) == True:
                points_var.append(pt)

        points_bcr = np.array(points_bcr_).reshape((-1,num))
        points_var = np.array(points_var).reshape((-1,num))
        rbcr = np.matmul(points_bcr,rew)
        rvar = np.matmul(points_var,rew)
        rets_bcr.append(np.min(rbcr))
        try:
            rets_var.append(np.min(rvar))
        except:
            print(num)
    rets_bcr = np.array(rets_bcr)
    rets_var = np.array(rets_var)
    percentage = np.abs(rets_var-rets_bcr)/(rets_bcr)
    x = np.arange(len(percentage))
    plt.plot(x,percentage)
    plt.figure(figsize=(7, 6))


    plt.xlabel("#States")
    plt.ylabel("Percentage change in returns")
    plt.savefig('rets_graph.pdf',type='pdf')


def compute_worst_returns(mu,Sigma,rsquared,value):

    print("mu",mu)
    print("Sigma",Sigma)
    print("rsquaed",rsquared)
    print("value",value)

    nrows, ncols = Sigma.shape
    m = gp.Model()
    x = m.addMVar(nrows, lb=0, ub=gp.GRB.INFINITY, name="x")
    temp1 = m.addMVar(nrows, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp1")
    temp2 = m.addMVar(nrows, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp2")
    temp3 = m.addMVar(nrows, lb=0, ub=gp.GRB.INFINITY, name="temp2")

    for idx in range(nrows):
        m.addConstr(temp1[idx] == (x[idx] - mu[idx]), name="c2" + str(idx))

    for idx in range(ncols):
        m.addConstr(temp2[idx] == gp.quicksum(temp1[jdx]*Sigma[jdx,idx] for jdx in range(nrows)), "c3" + str(idx))

    for idx in range(nrows):
        m.addConstr(temp3[idx] == temp2[idx] * temp1[idx], "c4" + str(idx))

    m.addConstr(gp.quicksum(temp3[idx] for idx in range(nrows)) <= rsquared, name="c5")

    # for idx in range(self.n):
    #     m.addConstr(x[idx] >= 0, name="c6" + str(idx))

    m.params.NonConvex = 2
    m.setObjective(gp.quicksum(value[idx] * x[idx] for idx in range(nrows)), gp.GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)

    m.optimize()
    # for v in m.getVars():
    #     print(f"{v.VarName} = {v.X}")
    vals = np.round(np.array(x.X), 2)

    m.dispose()
    del m
    return vals

def compute_returns(rewards,radius,mean):

    s = mean.shape[0]
    m = gp.Model()
    x = m.addMVar(s, lb=0, ub=gp.GRB.INFINITY, name="x")
    temp1 = m.addMVar(s, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp")
    temp2 = m.addMVar(s, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="temp1")

    for jdx in range(s):
        m.addConstr(temp1[jdx] == x[jdx]-mean[jdx], "c"  + "_" + str(jdx) )
        m.addConstr(temp2[jdx] == gp.abs_(temp1[jdx]), "c"  + "_" + str(jdx) )
        m.addConstr(x[jdx]>=0, "c1"  + "_" + str(jdx) )



    m.addConstr(gp.quicksum(x[idx] for idx in range(s)) ==1,"c1")


    m.addConstr(gp.quicksum(temp2[idx] for idx in range(s)) <= radius,"r1")

    # m.params.NonConvex = 2
    m.setObjective(gp.quicksum(rewards[idx] * x[idx] for idx in range(s)), gp.GRB.MINIMIZE)
    m.setParam('OutputFlag', 1)

    m.optimize()
    # for v in m.getVars():
    #     print(f"{v.VarName} = {v.X}")
    vals = np.round(np.array(x.X), 5)


    m.dispose()
    del m
    return vals


def compute_radius(samples,rewards, delta):
    alpha1 = 1-delta
    n = samples.shape[0]
    rewards = rewards.flatten()
    mean = np.mean(samples,axis=0)
    dist =np.sum(np.abs(samples - mean.reshape(1,-1)),axis=1)

    indices = np.argsort(dist)

    k = int(alpha1*n)
    index = indices[k-1]
    idxs = indices[:k]
    pvals = samples[idxs,:]
    returns_all = np.matmul(pvals,rewards).flatten()
    print("mean")
    radius = dist[index]
    print("radius",radius)
    p_ = compute_returns(rewards,radius,mean)
    print("optimal p_",p_)
    print("rets",np.dot(p_,rewards))
    return radius





if __name__=='__main__':
    alpha = np.array([10, 10, 1])
    samples = dirichlet.rvs(alpha, size=100, random_state=10)
    r = np.array([0.25, 0.25, -1])

    compute_radius(samples, r, 0.2)



    rets = np.matmul(samples, r)
    k = 0.2 * samples.shape[0]
    ret = np.quantile(rets.flatten(), 0.2)

    print("ret 20",ret)
    plot_radii()
