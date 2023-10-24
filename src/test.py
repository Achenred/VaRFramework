from algorithms import *

if __name__=='__main__':
    args = parse()
    domain = args.env
    delta = float(args.delta)

    method = args.method
    seed = args.seed
    nsa = args.nsa


    seeds=[0,1,2,3,4,5,6,7,8,9]
    # seeds=[1]
    for nsa_ in [10,50,100]:
        for seed in seeds:
            seed = int(seed)
            deltas=[0.3,0.15,0.05]
            for delta in deltas:
                if args.env == "inventory":
                    nsa = 4
                    continue
                else:
                    nsa=nsa_
                load_all(domain,delta=delta, seed=seed, nsa=args.nsa)

