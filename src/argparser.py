import argparse


def parse():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Implementation of MPO on gym environments')

    parser.add_argument('--env', type=str, default="population",
                        help='riverswim,inventory,population,population_small')

    parser.add_argument('--method', type=str, default="l1_bcr",
                        help='naive_bcr_l1,naive_bcr_linf,weighted_bcr_l1,weighted_bcr_linf,var')

    parser.add_argument('--delta', type=float, default=0.2,
                        help='discount parameter')

    parser.add_argument('--seed', type=float, default=0,
                        help='seed')
    parser.add_argument('--nsa', type=float, default=4,
                        help='nsa')

    return parser.parse_args()
