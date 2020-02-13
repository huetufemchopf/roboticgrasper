from __future__ import absolute_import
import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Short Project Grasping arm with Reinforcement Learning')

    # Datasets parameters
    parser.add_argument('--model_dir', type=str, default='./roboticgrasper/log',help="place where it saves model")

    parser.add_argument('--random_seed', type=float, default=5,help="random seed")
    parser.add_argument('--gym_env', type=str, default='possensor',help="Gym Environment: either possensor, rgbd, rgb, rgbdsparse, multi")
    parser.add_argument('--algorithm', type=str, default='DQN',help="currently DQN and DDPG supported. If DDPG, but then the Gym environment has to be non-discrete")
    parser.add_argument('--discrete', type=bool, default=True,help="if DQN discrete is true, if DDPG it is false")

    parser.add_argument('--lr', type=float, default=0.0001,help="learning rate")
    parser.add_argument('--discount', type=float, default=0.99,help="discount factor")
    parser.add_argument('--renders', type=bool, default=True,help="discount factor")
    parser.add_argument('--timesteps', type=int, default=10000,help="time steps")
    parser.add_argument('--logdir', type=str, default='logdir_DQN_rgbd_07-02/best_model.zip',help="path to stored files")




    args = parser.parse_args()

    return args


