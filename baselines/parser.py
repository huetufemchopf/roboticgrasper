from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Short Project Grasping arm with Reinforcement Learning')

    # Datasets parameters
    parser.add_argument('--model_dir', type=str, default='./roboticgrasper/log',help="place where it saves model")

    parser.add_argument('--random_seed', type=str, default=5,help="random seed")
    parser.add_argument('--gym_env', type=str, default='possensor',help="Gym Environment: either possensor, rgbd or rgb")
    parser.add_argument('--algorithm', type=str, default='DQN',help="currently DQN and DDPG supported")
    parser.add_argument('--modelname', type=str, default='tm700_ddpg_possensor_bestmodel.pkl',help="currently DQN and DDPG supported")



    args = parser.parse_args()

    return args


