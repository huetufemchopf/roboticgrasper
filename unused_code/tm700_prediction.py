import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from bullet.tm700_possensor_Gym import tm700_possensor_gym
from bullet.tm700_rgbd_Gym import tm700_rgbd_gym
from bullet.tm700_rgb_Gym import tm700_rgb_gym
from stable_baselines import DQN, PPO2, DDPG
from baselines.helpers import evaluate
import baselines.parser as parser
import time
from datetime import date
import baselines.helpers as helpers
from tensorflow.random import set_random_seed
import tensorflow as tf
import numpy as np
from baselines.helpers import record_gif
from stable_baselines import results_plotter
#################### PARAMETERS
args = parser.arg_parse()
np.random.seed(0)
set_random_seed(args.random_seed)

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )
sess = tf.Session(config=config)

start = time.time()
ENVIRONMENT = 'possensor'
MODEL = 'DDPG'
DISCRETE = False
DATE = date.today().strftime("%d-%m")
RENDERS = True
MODELPATH = None
# MODELNAME = 'trainedmodel_DQN_rgbd_06-02.pkl' #
MODELNAME = "tm700_ddpg_possensor_bestmodel.pkl"

################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'rgbd':
  env = tm700_rgbd_gym(renders=RENDERS, isDiscrete=DISCRETE)


if ENVIRONMENT == 'rgb':
  env = tm700_rgb_gym(renders=RENDERS, isDiscrete=DISCRETE)

if ENVIRONMENT == 'possensor':
  env = tm700_possensor_gym(renders=RENDERS, isDiscrete=DISCRETE)

if MODEL == 'DQN':

  model = DQN.load(MODELNAME, env=env)

if MODEL == 'DDPG':

  model = DDPG.load(MODELNAME, env=env)
  # model.predict(env, deterministic=True)


if MODEL == 'DDPG':
  model = DDPG.load(MODELNAME, env=env)


########## get baseline performance
#
# obs = model.env.observation_space.sample()
# baseline = model.predict(obs, deterministic=True)
# print(baseline)

# record_gif(model, env,  "%s_%s_%s.gif" % (MODEL, ENVIRONMENT, DATE) )





# runsimulation(model, env, 2000)

# evaluate(model, 20)

