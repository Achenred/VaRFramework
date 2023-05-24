import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('src/plot_style2.txt')
import seaborn as sns
np.random.seed(10)

from algorithms import *


class Evaluator():
    def __init__(self,env,dir="logs/"):
        self.env = env
        self.markers=['.','o','v','*','+','x','d','s','<']
        path = "Domains/" +  env
        initial = load_initial(path)
        gamma = load_parameters(path)
        self.psamples, self.rewards = load_train_data(path)
        self.psamples_test, rewards_test = load_test_data(path)
        self.initial = initial
        true_model, reward = load_true(path)
        self.mdp = MDP(initial, self.psamples.shape[1], self.psamples.shape[2], reward, true_model, gamma)
        self.dir = "new_plots/" + env
        self.labels = ["var","naive_bcr_l1","naive_bcr_linf","weighted_bcr_l1","weighted_bcr_linf","mean_model","asymptoticvar"]
        self.flabels = ["var","naive_bcr_l1","naive_bcr_linf","weighted_bcr_l1","weighted_bcr_linf","asymptoticvar"]

        self.map = {"weighted_bcr_l1":"WBCR $l_1$","naive_bcr_l1":"BCR $l_1$","weighted_bcr_linf":"WBCR $l_{\infty}$","var":"VaR","naive_bcr_linf":"BCR $l_{\infty}$","mean_model": "Mean","asymptoticvar": "VaRN"}
        self.deltas = [0.2,0.15,0.1,0.05,0.01]
        self.fdeltas = [0.15,0.1,0.05,0.01]

        self.initials = self.load_all_initial()



    def compute_all(self, dir="logs/"):
        res = {}
        for file in os.listdir(dir):
            if self.env not in file:
                continue
            try:
                delta = float(file.split("-")[1])
                domain = file.split("-")[0]
                name =domain + "_" + str(delta) + "_"
                new_dir = "all_logs/"
                if os.path.exists(new_dir) is False:
                    os.mkdir(new_dir)
                path = dir + file + "/"
                new_path = new_dir + file + "/"
                wbcr_l1 = np.load(path + "l1_optimized_robust_policy.npy")
                bcr_l1 = np.load(path + "l1_naive_robust_policy.npy")
                wbcr_linf = np.load(path + "linf_optimized_robust_policy.npy")
                bcr_linf = np.load(path + "linf_naive_robust_policy.npy")
                var_robust = np.load(path + "var_policy.npy")
                asymptoticvar_robust = np.load(path + "asymptoticvar_policy.npy")



                method = "var"
                train_ret1 = self.evaluate_all(var_robust, self.rewards, new_path + method, test=True)

                method = "asymptoticvar"
                train_ret8 = self.evaluate_all(asymptoticvar_robust, self.rewards, new_path + method, test=True)

                method = "l1_naive_robust"
                train_ret2 = self.evaluate_all(bcr_l1, self.rewards, new_path + method, test=True)
                method = "l1_optimized_robus"
                train_ret3 = self.evaluate_all(wbcr_l1, self.rewards, new_path + method, test=True)
                method = "linf_naive_robust"
                test_ret4 = self.evaluate_all(bcr_linf, self.rewards, new_path + method, test=True)
                method = "linf_optimized_robust"
                test_ret5 = self.evaluate_all(wbcr_linf, self.rewards, new_path + method, test=True)
                print("VaR percentile Returns", np.percentile(train_ret1, delta))
                print("Asymptotic VaR percentile Returns", np.percentile(train_ret8, delta))

                print("Naive bcr percentile Returns", np.percentile(train_ret2, delta))
                print("Optimized bcr percentile Returns", np.percentile(train_ret3, delta))
                print("naive bcr percentile Returns linf", np.percentile(test_ret4, delta))
                print("Optimized bcr percentile Returns linf", np.percentile(test_ret5, delta))

                method = "var"
                train_ret1 = self.evaluate_all(var_robust, self.rewards, new_path + method, test=False)

                method = "asymptoticvar"
                train_ret8 = self.evaluate_all(asymptoticvar_robust, self.rewards, new_path + method, test=False)

                method = "l1_naive_robust"
                train_ret2 = self.evaluate_all(bcr_l1, self.rewards, new_path + method, test=False)
                method = "l1_optimized_robus"
                train_ret3 = self.evaluate_all(wbcr_l1, self.rewards, new_path + method, test=False)
                method = "linf_naive_robust"
                test_ret4 = self.evaluate_all(bcr_linf, self.rewards, new_path + method, test=False)
                method = "linf_optimized_robust"
                test_ret5 = self.evaluate_all(wbcr_linf, self.rewards, new_path + method, test=False)
                print("VaR percentile Returns", np.percentile(train_ret1, delta))
                print("Asymptotic VaR percentile Returns", np.percentile(train_ret8, delta))

                print("Naive bcr percentile Returns", np.percentile(train_ret2, delta))
                print("Optimized bcr percentile Returns", np.percentile(train_ret3, delta))
                print("naive bcr percentile Returns linf", np.percentile(test_ret4, delta))
                print("Optimized bcr percentile Returns linf", np.percentile(test_ret5, delta))

            except:
                continue

    def load_all_initial(self):
        initials={}
        envs = ['riverswim','inventory','population','population_small']
        for env in envs:
            path = "Domains/" +  env
            init = load_initial(path)
            initials[env]=init
        return initials


    def compute_value_rets(self, dir="logs/"):
        res = {}
        values={}
        for file in os.listdir(dir):
            # if self.env not in file:
            #     continue
            # try:
                delta = float(file.split("-")[1])
                domain = file.split("-")[0]
                name = domain + "_" + str(delta) + "_"
                new_dir = "all_logs/"
                if os.path.exists(new_dir) is False:
                    os.mkdir(new_dir)
                path = dir + file + "/"
                new_path = new_dir + file + "/"
                wbcr_l1 = np.load(path + "weighted_bcr_l1_rets.npy")
                bcr_l1 = np.load(path + "naive_bcr_l1_rets.npy")
                wbcr_linf = np.load(path + "weighted_bcr_linf_rets.npy")
                bcr_linf = np.load(path + "naive_bcr_linf_rets.npy")
                var_robust = np.load(path + "var_rets.npy")
                mean = np.load(path + "mean_model_rets.npy")
                true = np.load(path + "true_model_rets.npy")
                asymptoticvar_robust = np.load(path + "asymptoticvar_rets.npy")

                try:
                    init = self.initials[domain]
                except:
                    for key in list(self.initials.keys()):
                        if domain in key:
                            init = self.initials[key]

                wbcr_l1_val = np.load(path + "l1_optimized_robust_value.npy")
                wbcr_l1_val = np.dot(init,wbcr_l1_val)
                bcr_l1_val = np.load(path + "l1_naive_robust_value.npy")
                bcr_l1_val = np.dot(init, bcr_l1_val)
                wbcr_linf_val = np.load(path + "linf_optimized_robust_value.npy")
                wbcr_linf_val = np.dot(init, wbcr_linf_val)
                bcr_linf_val = np.load(path + "linf_naive_robust_value.npy")
                bcr_linf_val = np.dot(init, bcr_linf_val)
                var_robust_val = np.load(path + "var_value.npy")
                var_robust_val = np.dot(init, var_robust_val)
                asymptoticvar_robust_val = np.load(path + "asymptoticvar_value.npy")
                asymptoticvar_robust_val = np.dot(init, asymptoticvar_robust_val)
                mean_val = np.load(path + "_mean_value.npy")
                mean_val = np.dot(init, mean_val)
                # true_val = np.load(path + "_true_value.npy")
                # true_val = np.dot(self.mdp.p0, true_val )

                if res.get(domain) is None:
                    res[domain] = {}
                if res[domain].get(delta) is None:
                    res[domain][delta] = {}
                res[domain][delta]["weighted_bcr_l1"] = wbcr_l1
                res[domain][delta]["naive_bcr_l1"] = bcr_l1
                res[domain][delta]["weighted_bcr_linf"] = wbcr_linf
                res[domain][delta]["naive_bcr_linf"] = bcr_linf
                res[domain][delta]["var"] = var_robust
                res[domain][delta]["asymptoticvar"] = asymptoticvar_robust

                res[domain][delta]["mean_model"] = mean

                if values.get(domain) is None:
                    values[domain] = {}
                if values[domain].get(delta) is None:
                    values[domain][delta] = {}
                values[domain][delta]["weighted_bcr_l1"] = wbcr_l1_val
                values[domain][delta]["naive_bcr_l1"] = bcr_l1_val
                values[domain][delta]["weighted_bcr_linf"] = wbcr_linf_val
                values[domain][delta]["naive_bcr_linf"] = bcr_linf_val
                values[domain][delta]["var"] = var_robust_val
                values[domain][delta]["asymptoticvar"] = asymptoticvar_robust_val

                values[domain][delta]["mean_model"] = mean_val



            # except:
            #     pass
        return res, values





    def load_all(self,dir="logs/"):
        res={}
        true_res={}
        for sub_dir in os.listdir(dir):
            domain = sub_dir.split("-")[0]
            delta = float(sub_dir.split("-")[1])
            path = dir + sub_dir+"/"

            for file in os.listdir(path):
                method = "_".join(file.split("_")[:-2])
                if "_policy" in path:
                    continue
                path_ = path + file

                if "true_returns" in path_:
                    rets = np.load(path_)

                    if true_res.get(domain) is None:
                        true_res[domain]={}
                    if true_res[domain].get(delta) is None:
                        true_res[domain][delta]=rets
                else:


                    if "test" in file:
                        type="test"
                    else:
                        type = "train"
                    if res.get(domain) is None:
                        res[domain]={}

                    if res[domain].get(type) is None:
                        res[domain][type]={}

                    if res[domain][type].get(delta) is None:
                        res[domain][type][delta]={}
                    if res[domain][type][delta].get(method) is None:
                        res[domain][type][delta][method]=[]

                    rets = np.load(path_)
                    res[domain][type][delta][method]=rets


        return res, true_res



    def load_policies(self,dir="logs/"):
        res = {}
        for file in os.listdir(dir):

            try:
                delta = float(file.split("-")[1])
                domain = file.split("-")[0]
                name = domain + "_" + str(delta) + "_"
                new_dir = "logs/"
                if os.path.exists(new_dir) is False:
                    os.mkdir(new_dir)
                path = dir + file + "/"
                new_path = new_dir + file + "/"
                wbcr_l1 = np.load(path + "l1_optimized_robust_policy.npy")
                bcr_l1 = np.load(path + "l1_naive_robust_policy.npy")
                wbcr_linf = np.load(path + "linf_optimized_robust_policy.npy")
                bcr_linf = np.load(path + "linf_naive_robust_policy.npy")
                var_robust = np.load(path + "var_policy.npy")
                asymptoticvar_robust = np.load(path + "asymptoticvar_policy.npy")

                mean_policy = np.load(path + "_mean_policy.npy")
                if res.get(domain) is None:
                    res[domain]={}
                if res[domain].get(delta) is None:
                    res[domain][delta]={}
                res[domain][delta]["weighted_bcr_l1"]=wbcr_l1
                res[domain][delta]["naive_bcr_l1"] = bcr_l1
                res[domain][delta]["weighted_bcr_linf"]=wbcr_linf
                res[domain][delta]["naive_bcr_linf"]=bcr_linf
                res[domain][delta]["var"]=var_robust
                res[domain][delta]["asymptoticvar"]=asymptoticvar_robust

                res[domain][delta]["mean"]= mean_policy

            except:
                pass

        return res

    def evaluate_all(self,pi,rew, path, test=True):
        if test == True:
            psamples = self.psamples_test
        else:
            psamples = self.psamples
        self.ns = psamples.shape[1]
        self.na = psamples.shape[2]
        num = psamples.shape[0]
        returns = []
        I = np.eye(self.ns)
        for idx in range(num):
            pi_p_temp = pi.reshape((self.ns,self.na,1)) * psamples[idx,:,:,:]
            pi_p = np.sum(pi_p_temp,axis=1)
            pi_p_inverse = np.linalg.inv(I - self.mdp.gamma *pi_p)
            r = np.sum(np.sum(pi_p_temp* rew,-1),-1)
            ret = np.matmul(pi_p_inverse,r)
            returns.append(np.dot(self.mdp.p0, ret))
        if test is True:
            fname = path + "_test_rets.npy"
        else:
            fname = path +"_train_rets.npy"

        np.save(fname,np.array(returns))
        return returns


    def plot_policy(self,policy,name):

        plt.clf()
        plt.style.use('src/plot_style2.txt')

        fig = plt.gcf()
        plt.figure(figsize=(10.7,7))
        # cmap = plt.cm.Blues
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=1.0)
        # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        # sm.set_array([])
        # ax = plt.gca()
        # fig.colorbar(sm,ax)
        # plt.matshow(policy, cmap=cmap, vmin=0.0, vmax=1.0,alpha=0.7)
        sns.heatmap(policy, cmap='magma', linecolor='white', linewidths=1,xticklabels=np.arange(policy.shape[1]))

        # plt.xticks(np.arange(policy.shape[1]))
        plt.xlabel('actions')
        plt.ylabel('states')

        plt.savefig("new_plots/"+name + ".pdf")
        plt.close()
        plt.close(fig)


    def plot_percentile_returns(self,returns, true_returns, fname):
        plt.clf()
        plt.style.use('src/plot_style2.txt')

        plt.figure(figsize=(10.7,7))
        # plt.axhline(y=true_returns, linestyle='--',label="true returns",color='magenta')

        self.deltas = np.array(self.deltas)
        methods = list(self.map.keys())
        results = {}
        line = ""
        line2=""
        # print("Fname", fname)
        idx=0
        for method in self.flabels:
            results[method]=[]
            name = self.map[method]
            vals=[]
            for delta in self.fdeltas:
                ret = returns[delta][method]
                val = np.quantile(ret,delta)
                val = np.round(val,2)
                vals.append(val)
                results[method].append((delta,val))

                if (delta== 0.15):
                    line += str(val) + " & "
                if (delta== 0.05):
                    line2 += str(val) + " & "
            vals = np.array(vals)
            print(fname + "-" + str(method), results[method])

            plt.plot(self.fdeltas,vals,label=self.map[method],marker=self.markers[idx],linewidth=2, markersize=4,alpha=0.7)
            idx+=1

        ax = plt.gca()
        # plt.legend(loc='upper right')
        ax.legend(loc='upper right', bbox_to_anchor=(1.12, 0.9),
                  ncol=1)
        # ax.legend(bbox_to_anchor=(1.0, 1.0))

        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, ncol=3)
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                  box.width, box.height * 0.9])

        # Put a legend below current axis
        # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
        #           fancybox=True, shadow=True, ncol=5)

        plt.xlabel('quantile δ')
        plt.ylabel('Returns')
        plt.savefig("new_plots/"+fname+"_percentile.pdf")
        plt.close()
        # print(fname + "-line1-" + line)
        # print(fname + "-line2-" + line2)

    def plot_true_returns(self,returns, fname):
        plt.clf()
        plt.style.use('src/plot_style2.txt')

        plt.figure(figsize=(10.7,7))

        self.deltas = np.array(self.deltas)
        methods = list(self.map.keys())
        results = {}
        line = ""
        # print("Fname", fname)
        idx = 0
        for method in self.flabels:

            results[method]=[]
            name = self.map[method]
            vals=[]
            for delta in self.fdeltas:
                ret = returns[delta][method]
                # val = np.quantile(ret,delta)
                # val = np.round(val,2)
                vals.append(np.round(ret,2))
                results[method].append((delta,val))
                if delta== 0.15 or delta==0.05:
                    line += str(val) + " & "
            vals = np.array(vals)
            # print(fname + "-" + str(method), results[method])

            plt.plot(self.fdeltas,vals,label=self.map[method],marker=self.markers[idx],linewidth=2, markersize=4,alpha=0.7)
            idx+=1
        plt.legend(loc='upper right')
        plt.xlabel('quantile δ')
        plt.ylabel('Returns on true model')
        plt.savefig("new_plots/"+fname+"_true_returns.pdf")
        plt.close()
        # print(fname + "-" + line)
        plt.clf()

    def plot_lower_bound(self,true_val,returns,  fname):
        plt.clf()
        plt.style.use('src/plot_style3.txt')

        plt.figure(figsize=(10.7,7))
        plt.axhline(y=true_val, linestyle='--',label="true returns",color='magenta')

        self.deltas = np.array(self.deltas)
        methods = list(self.map.keys())
        results = {}
        line = ""
        # print("Fname", fname)
        idx=0
        for method in self.flabels:
            results[method]=[]
            name = self.map[method]
            vals=[]
            for delta in self.fdeltas:
                ret = returns[delta][method]
                # val = np.quantile(ret,delta)
                # val = np.round(val,2)
                vals.append(ret)
                results[method].append((delta,val))
                if delta== 0.15 or delta==0.05:
                    line += str(val) + " & "
            vals = np.array(vals)
            # print(fname + "-" + str(method), results[method])

            plt.plot(self.fdeltas,vals,label=self.map[method],marker=self.markers[idx],linewidth=2, markersize=4,alpha=0.7)
            idx+=1
        plt.legend(loc='upper right')
        plt.xlabel('quantile δ')
        plt.ylabel('Lower bound v')
        plt.savefig("new_plots/"+fname+"_lower_bound.pdf")
        plt.close()
        # print(fname + "-" + line)
        plt.clf()


    def plot_bar_percentile_returns(self,tr_rets,te_rets, true_ret, fname,delta=0.05):
        plt.clf()
        plt.style.use('src/plot_style2.txt')

        plt.figure(figsize=(10,9),dpi=60)

        self.deltas = np.array(self.deltas)
        methods =list(self.map.keys())
        results = {}
        line = ""

        vals = []
        # vals.append(true_ret)
        vals_test = []
        # vals_test.append(true_ret)

        # print("Fname", fname)
        labels_str = []
        idx=0
        for method in self.flabels:
            results[method]=[]
            name = self.map[method]
            ret =tr_rets[delta][method]
            ret_test = te_rets[delta][method]
            val = np.quantile(ret, delta)
            val = np.round(val, 2)
            val_t = np.quantile(ret_test, delta)
            val_t = np.round(val_t, 2)
            vals.append(val)
            vals_test.append(val_t)
            labels_str.append(self.map[method])

        barWidth = 0.25

        br1 = np.arange(len(self.flabels))
        br2 = [x + barWidth for x in br1]
        vals = np.array(vals)
        vals_test = np.array(vals_test)
        # Make the plot

        plt.bar(br1, vals, color='r', width=barWidth,
                edgecolor='grey', label='train',alpha=0.5)
        plt.bar(br2,vals_test,  color='b', width=barWidth,
                edgecolor='grey', label='test',alpha=0.5)
        plt.xticks([r + barWidth for r in range(len(self.flabels))],
                   labels_str,rotation=30,fontsize=20)


        # plt.plot(self.deltas,vals,label=self.map[method],marker='o',linewidth=2, markersize=4,alpha=0.7)

        plt.legend(loc='upper right',fontsize=14)
        plt.xlabel('Methods')
        plt.ylabel("δ="+str(delta)+ " quantile" +' returns')
        plt.savefig("new_plots/"+fname+"_bar.pdf")
        plt.close()
        # print(fname + "-" + line)



    def plot_returns(self,returns,fname):
        plt.clf()
        plt.style.use('src/plot_style2.txt')

        self.deltas = np.array(self.deltas)
        methods = list(self.map.keys())
        plt.figure(figsize=(7,6))
        idx=0
        for method in methods:
            for delta in self.deltas:
                plt.clf()
                ret = returns[delta][method]
                plt.hist(ret, bins=100, alpha=0.5, label=self.map[method],density=True)
                plt.xlabel("Returns", size=14)
                plt.ylabel("Frequency", size=14)
                plt.legend(loc='upper right')

        plt.legend(loc='upper right')

        plt.savefig("new_plots/"+fname + "_percentile.pdf")
        plt.close()





if __name__=='__main__':
    env="riverswim"
    eval = Evaluator(env)
    res, true_res = eval.load_all()
    plot_dir = "new_plots/"
    for domain, val in res.items():
        true_val = list(true_res[domain].values())[0][0]
        for type, rets in val.items():
            name = domain  + "-"+ str(type)
            # eval.plot_returns(rets,name)

            eval.plot_percentile_returns(rets, true_val,name)
        eval.plot_bar_percentile_returns(res[domain]['train'],res[domain]['test'],true_val,domain)

    res, values = eval.compute_value_rets()

    for domain, val in res.items():
            true_val = list(true_res[domain].values())[0][0]

        # true_val = list(true_res[domain].values())[0][0]
        # for type, rets in val.items():
            name = domain
            # eval.plot_returns(rets,name)

            eval.plot_true_returns(val, name)

    for domain, val in values.items():
            true_val = list(true_res[domain].values())[0][0]

        # true_val = list(true_res[domain].values())[0][0]
        # for type, rets in val.items():
            name = domain
            # eval.plot_returns(rets,name)

            eval.plot_lower_bound(true_val,val, name)


    policies = eval.load_policies()
    for domain, val in policies.items():

        for delta, method_val in val.items():
            for method, val in method_val.items():
                plt.clf()
                path = "new_plots/"+ domain+"/"
                if os.path.exists(path) is False:
                    os.mkdir(path)
                name = domain + "/" + str(delta) + "-" + method
                policy = policies[domain][delta][method]
                eval.plot_policy(policy,name)

