
#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import parser
import gym
from bullet.tm700_diverse_object_gym_env import tm700DiverseObjectEnv
from bullet.tm700GymEnv import tm700GymEnv2
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv
from stable_baselines import deepq
from stable_baselines import DQN, PPO2, DDPG, HER
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
import datetime
# from stable_baselines_test.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import DDPGPolicy,MlpPolicy
import time
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines_test.common.policies import MlpPolicy
import numpy as np
from stable_baselines.common.vec_env import VecVideoRecorder



def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    successratio = 0
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            if reward > 950:
                successratio +=1

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    successratio = successratio/num_episodes
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes, "Success ratio:", successratio)

    return mean_episode_reward


def record_video(env_id, model, video_length=500, prefix='', video_folder='videos/'):
  """
  :param env_id: (str)
  :param model: (RL model)
  :param video_length: (int)
  :param prefix: (str)
  :param video_folder: (str)
  """
  eval_env = DummyVecEnv([lambda: gym.make(env_id)])
  # Start the video at step=0 and record 500 steps
  eval_env = VecVideoRecorder(env, video_folder=video_folder,
                              record_video_trigger=lambda step: step == 0, video_length=video_length,
                              name_prefix=prefix)

  obs = eval_env.reset()
  for _ in range(video_length):
    action, _ = model.predict(obs)
    obs, _, _, _ = eval_env.step(action)

  # Close the video recorder
  eval_env.close()

def savemodel(model, MODEL, ENVIRONMENT, DATE):

    model.save("trainedmodel_%s_%s_%s.pkl" % MODEL, ENVIRONMENT, DATE)

    pass

if __name__ == '__main__':

    ################################## Filter tensorflow version warnings
    import os

    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import warnings

    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging

    tf.get_logger().setLevel(logging.ERROR)

    args=parser.arg_parse()

    # Create save dir
    model_dir = args.model_dir

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)


    ############## load environment

    env = tm700GymEnv2(renders=False, isDiscrete=True)
    # env = gym.make('CartPole-v1')
    # vectorized environments allow to easily multiprocess training
    # we demonstrate its usefulness in the next examples
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run


    ############ MODELS

    model = PPO2('MlpPolicy', 'Pendulum-v0', verbose=0).learn(8000)
    # The model will be saved under PPO2_tutorial.zip
    # ddpg_model = DDPG(MlpPolicy, env, verbose=1, param_noise=None, random_exploration=0.1)
    kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}  #DQN + Prioritized Experience Replay + Double Q-Learning + Dueling
    dqn_model = DQN('MlpPolicy', env, verbose=1, **kwargs)


    ############ learn & evaluate

    mean_reward_before_train = evaluate(dqn_model, num_episodes=50)
    dqn_model.learn(10000)



    # print("loaded", loaded_model.predict(obs, deterministic=True))

    mean_reward_after_train = evaluate(dqn_model, num_episodes=50)


    ################# EVALUATE with loaded model

    # model = DDPG.load("tm_test_model.pkl", env=env)
    # sample an observation from the environment
    # obs = model.env.observation_space.sample()
    # print("loaded", model.predict(obs, deterministic=True))



    ################### SAVE MODEL

    # Check prediction before saving
    # print("pre saved", model.predict(obs, deterministic=True))

    model.save(args.model_dir + "/tutorial")
