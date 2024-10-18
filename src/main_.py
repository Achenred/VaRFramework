#
# from algorithms import *
#
# if __name__=='__main__':
#     args = parse()
#     domain = args.env
#     delta = float(args.delta)
#
#
#     method = args.method
#     seed = args.seed
#     nsa = args.nsa
#     if args.env == "inventory":
#         nsa = 4
#
#     seeds=[0,1,2,3,4,5,6,7,8,9]
#     seed = int(seed)
#     # seeds=[1]
#     for seed in seeds:
#         deltas=[0.3,0.15,0.05]
#         for delta in deltas:
#             load_all(domain,delta=delta, seed=seed, nsa=args.nsa)
#
#

# Example usage:

S = 5  # Number of states
A = 2  # Number of actions
np.random.seed(0)
P = np.random.rand(S, A, S)
P /= P.sum(axis=2, keepdims=True)

R = np.random.rand(S, A)

gamma = 0.99
R *= (1 - gamma)

risk_tau = 0.75
model = DynamicRiskBasedModel(S, A, P, R, gamma, risk_tau)
v, policy = model.dynamic_risk_iteration()

print("Value Function:", v)
print("Optimal Policy:", policy)

