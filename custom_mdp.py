import os
import sys
import numpy as np
from scipy.stats import dirichlet


class MDP():
    def __init__(self, ns, na, alphas, rewards, initial, mdp_name):
        self.ns = ns
        self.na = na

        self.alphas = alphas
        self.rewards = rewards
        self.initial = initial
        self.mdp_dir = mdp_name + "/"


    def sample(self, size):
        train_samples = dirichlet.rvs(self.alphas, size=size, random_state=1)
        test_samples = dirichlet.rvs(self.alphas, size=size, random_state=1)
        mean = dirichlet.mean(self.alphas)
        init = open(self.mdp_dir + "initial.csv",'r+')
        init.write("idstate,probability\n")
        for idx in range(self.ns):
            init.write(str(idx)+ ","+str(self.initial[idx])+"\n")
        init.close()
        trainer = open(self.mdp_dir + "training.csv",'r+')
        trainer.write("idstatefrom,idaction,idstateto,idoutcome,probability,reward\n")
        shape = train_samples.shape
        for idx in range(shape[0]):
            for jdx in range(shape[1]):
                for kdx in range(shape[2]):
                    for ldx in range(shape[3]):
                        probs = train_samples[idx,jdx,kdx,ldx]
                        rew = self.rewards[idx,jdx,kdx]
                        res = str(jdx) + "," + str(kdx)+ ","+ str(ldx)+","+ str(idx)+","+ str(np.round(probs))+","+ str(rew)+"\n"
                        trainer.write(res+"\n")

        trainer = open(self.mdp_dir + "training.csv", 'r+')
        trainer.write("idstatefrom,idaction,idstateto,idoutcome,probability,reward\n")
        shape = train_samples.shape
        for idx in range(shape[0]):
            for jdx in range(shape[1]):
                for kdx in range(shape[2]):
                    for ldx in range(shape[3]):
                        probs = train_samples[idx, jdx, kdx, ldx]
                        rew = self.rewards[idx, jdx, kdx]
                        res = str(jdx) + "," + str(kdx) + "," + str(ldx) + "," + str(idx) + "," + str(
                            np.round(probs)) + "," + str(rew) + "\n"
                        trainer.write(res + "\n")

        trainer = open(self.mdp_dir + "training.csv", 'r+')
        trainer.write("idstatefrom,idaction,idstateto,idoutcome,probability,reward\n")
        shape = train_samples.shape
        for idx in range(shape[0]):
            for jdx in range(shape[1]):
                for kdx in range(shape[2]):
                    for ldx in range(shape[3]):
                        probs = train_samples[idx, jdx, kdx, ldx]
                        rew = self.rewards[idx, jdx, kdx]
                        res = str(jdx) + "," + str(kdx) + "," + str(ldx) + "," + str(idx) + "," + str(
                            np.round(probs)) + "," + str(rew) + "\n"
                        trainer.write(res + "\n")




