from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Short Project Grasping arm with Reinforcement Learning')

    # Datasets parameters
    parser.add_argument('--model_dir', type=str, default='./log',
                        help="place where it saves model")




    args = parser.parse_args()

    return args