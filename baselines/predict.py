import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)
import gym
from pybullet_envs.bullet.tm700GymEnv_TEST import tm700GymEnv2
from stable_baselines import DQN, PPO2, DDPG
from pybullet_envs.baselines.train_tm700_multivec import evaluate, record_video

#################### PARAMETERS

savedmodel = "tm_test_model_randomblocks.pkl"
env = tm700GymEnv2(renders=False, isDiscrete=False)
model = DDPG.load(savedmodel, env=env)




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

record_video(env, 500,)