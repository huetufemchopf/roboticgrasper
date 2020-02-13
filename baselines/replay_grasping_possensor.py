import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sisterdir = os.path.dirname(currentdir)
print(currentdir, parentdir)
os.sys.path.insert(0, parentdir)
os.sys.path.insert(0, sisterdir)

from bullet.tm700_rgb_Gym import tm700_rgb_gym
from unused_code.tm700_rgbdnormalized_Gym import tm700_rgbd_gym
from bullet.tm700_rgbd_multiobject_Gym import tm700_rgbdmultiobj_gym
from bullet.tm700_possensor_Gym import tm700_possensor_gym
from stable_baselines import DQN, DDPG
from baselines.helpers import evaluate
import baselines.parser as parser
import time
from datetime import date

#################### PARAMETERS
args = parser.arg_parse()

start = time.time()
ENVIRONMENT = args.gym_env
MODEL = args.algorithm
DISCRETE = args.discrete
DATE = date.today().strftime("%d-%m")
RENDERS = args.renders
MODELNAME = os.path.join(currentdir, args.logdir) #"./logdir_DQN_multi_11-02/best_model.zip"

################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'rgbd':
  env = tm700_rgbd_gym(renders=RENDERS, isDiscrete=DISCRETE)
if ENVIRONMENT == 'multi':
  env = tm700_rgbdmultiobj_gym(renders=RENDERS, isDiscrete=DISCRETE)

if ENVIRONMENT == 'rgb':
  env = tm700_rgb_gym(renders=RENDERS, isDiscrete=DISCRETE)


if ENVIRONMENT == 'possensor':
  env = tm700_possensor_gym(renders=RENDERS, isDiscrete=DISCRETE)

if MODEL == 'DQN':

  model = DQN.load(MODELNAME, env=env)


if MODEL == 'DDPG':
  model = DDPG.load(MODELNAME, env=env)


########## get baseline performance

# obs = model.env.observation_space.sample()
# baseline = model.predict(obs, deterministic=True)
# print(baseline)
#


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
# record_video(env, model)


########### EVALUATE MODEL

mean_episode_reward, successratio = evaluate(model, 100)
print('Success rate', successratio, MODEL, ENVIRONMENT)


