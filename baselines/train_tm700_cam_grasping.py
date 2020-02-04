#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from bullet.tm700CamGymEnv import tm700CamGymEnv
from bullet.newcamgarsping import tm700GraspingProceduralEnv
from bullet.tm700_diverse_object_gym_env import tm700DiverseObjectEnv
from bullet.tm700GymEnv import tm700GymEnv2

from stable_baselines import DQN, DDPG
from stable_baselines.deepq.policies import LnCnnPolicy
from baselines.helpers import evaluate,record_video, savemodel
from datetime import date
import time

import datetime

ENVIRONMENT = 'diverse'
MODEL = 'DQN'
DISCRETE = True
DATE = date.today().strftime("%d-%m")
RENDERS = True

################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'diverse':
  env = tm700DiverseObjectEnv(renders=RENDERS, isDiscrete=DISCRETE)

if ENVIRONMENT == 'normal':
  env = tm700GymEnv2(renders=RENDERS, isDiscrete=DISCRETE)

if MODEL == 'DQN':
  from stable_baselines.deepq.policies import LnCnnPolicy
  model = DQN(LnCnnPolicy, env, verbose=1, tensorboard_log="./tensorboard_%s_%s_%s/" % (MODEL, ENVIRONMENT, DATE) )

if MODEL == 'DDPG':
  from stable_baselines.ddpg.policies import MlpPolicy, LnCnnPolicy
  model = DDPG(LnCnnPolicy, env, verbose=1, random_exploration=0.1, tensorboard_log="./tensorboard_%s_%s_%s/" % (MODEL, ENVIRONMENT, DATE) )



################ TRAINING

model.learn(total_timesteps=500000)
print('save model')
savemodel(model, MODEL, ENVIRONMENT, DATE)



print('total time', time.time()-start)


