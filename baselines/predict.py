import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from bullet.tm700GymEnv import tm700GymEnv2
from bullet.tm700_diverse_object_gym_env import tm700DiverseObjectEnv
from stable_baselines import DQN, PPO2, DDPG
from baselines.helpers import evaluate

#################### PARAMETERS

#savedmodel = "tm700_cam_model.pkl"
#env = tm700DiverseObjectEnv(renders=True, isDiscrete=True)
#model = DQN.load(savedmodel, env=env)


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
exit()

