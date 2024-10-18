
from algorithms import *

if __name__=='__main__':
    # args = parse()
    # domain = args.env
    # delta = float(args.delta)
    # method = args.method
    # seed = args.seed
    # nsa = args.nsa
    # if args.env == "inventory":
    #     nsa = 4

    domain = "riverswim"
    delta = 0.05
    seed = 0
    nsa = 4

    seed = int(seed)
    load_all(domain,delta=delta, seed=seed, nsa= nsa)

