# code used from stable baselines tutorial: https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/2_gym_wrappers_saving_loading.ipynb#scrollTo=vBNFnN4Gd32g
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import parser
import gym
from bullet.tm700_rgbd_Gym import tm700_rgbd_gym
from bullet.tm700_possensor_Gym import tm700_possensor_gym
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
import imageio

import os

import numpy as np

def runsimulation(model, env, iterations):
    obs = env.reset()
    time_step_counter = 0
    iterations = iterations
    while time_step_counter < iterations:
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)
        # Assumption: eval conducted on single env only!
        time_step_counter +=1

        # time.sleep(0.1)
        if dones:
            obs = env.reset()

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

    return mean_episode_reward, successratio

def record_gif(model, env, name):
    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    for i in range(500):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _, _ = model.env.step(action)
        img = model.env.render(mode='rgb_array')

    imageio.mimsave(name, [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)


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


    name = "trainedmodel_%s_%s_%s.pkl" % (MODEL, ENVIRONMENT, DATE)
    print('save model as:', name)

    model.save(name)


class tm700Wrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    """

    def __init__(self, env):
        # Call the parent constructor, so we can access self.env later
        super(tm700Wrapper, self).__init__(env)

    def reset(self):
        """
        Reset the environment
        """
        obs = self.env.reset()
        return obs

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

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

    env = tm700_possensor_gym(renders=False, isDiscrete=True)
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
