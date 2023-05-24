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

    return parser.parse_args()
