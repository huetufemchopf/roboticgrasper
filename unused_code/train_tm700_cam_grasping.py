#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from bullet.tm700_rgbd_Gym import tm700_rgbd_gym
from bullet.tm700_possensor_Gym import tm700_possensor_gym

from stable_baselines import DQN, DDPG
from baselines.helpers import savemodel
from datetime import date
import time
import baselines.parser as parser

start = time.time()
ENVIRONMENT = 'diverse'
MODEL = 'DQN'
DISCRETE = True
DATE = date.today().strftime("%d-%m")
RENDERS = False

args = parser.arg_parse()
################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'rgbd':
  env = tm700_rgbd_gym(renders=RENDERS, isDiscrete=DISCRETE)

if ENVIRONMENT == 'possensor':
  env = tm700_possensor_gym(renders=RENDERS, isDiscrete=DISCRETE)

if MODEL == 'DQN':
  from stable_baselines.deepq.policies import LnCnnPolicy
  model = DQN(LnCnnPolicy, env, verbose=1, tensorboard_log="./tensorboard_%s_%s_%s/" % (MODEL, ENVIRONMENT, DATE) ,
              gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
              exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=False, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=False, n_cpu_tf_sess=None, _init_setup_model=True,
              policy_kwargs=None, full_tensorboard_log=False, seed=None)

if MODEL == 'DDPG':
  from stable_baselines.ddpg.policies import LnCnnPolicy
  model = DDPG(LnCnnPolicy, env, verbose=1, random_exploration=0.1, tensorboard_log="./tensorboard_%s_%s_%s/" % (MODEL, ENVIRONMENT, DATE) )



################ TRAINING

model.learn(total_timesteps=1000000)
print('save model')
savemodel(model, MODEL, ENVIRONMENT, DATE)



print('total time', time.time()-start)


