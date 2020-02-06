import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from bullet.tm700_possensor_Gym import tm700_possensor_gym
from bullet.tm700_rgbd_Gym import tm700_rgbd_gym
from stable_baselines import DQN, PPO2, DDPG
from baselines.helpers import evaluate
import baselines.parser as parser
import time
from datetime import date
import baselines.helpers as helpers

#################### PARAMETERS
args = parser.arg_parse()

start = time.time()
ENVIRONMENT = 'possensor'
MODEL = 'DDPG'
DISCRETE = False
DATE = date.today().strftime("%d-%m")
RENDERS = True
MODELPATH = None
MODELNAME = "tm700_ddpg_possensor_bestmodel.pkl"

################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'rgbd':
  env = tm700_rgbd_gym(renders=RENDERS, isDiscrete=DISCRETE)

if ENVIRONMENT == 'possensor':
  env = tm700_possensor_gym(renders=RENDERS, isDiscrete=DISCRETE)

if MODEL == 'DQN':

  model = DQN.load(MODELNAME, env=env)


if MODEL == 'DDPG':
  model = DDPG.load(MODELNAME, env=env)


########## get baseline performance

obs = model.env.observation_space.sample()
baseline = model.predict(obs, deterministic=True)
print(baseline)



########## run simulation

def runsimulation(model, env, iterations):
    obs = env.reset()
    time_step_counter = 0
    iterations = iterations
    while time_step_counter < iterations:
        action, _ = model.predict(obs)
        obs, rewards, dones, _ = env.step(action)  # Assumption: eval conducted on single env only!
        time_step_counter +=1

        # time.sleep(0.1)
        if dones:
            obs = env.reset()

# runsimulation(model, env, 2000)
evaluate(model, 100)

