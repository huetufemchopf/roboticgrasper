#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from bullet.tm700_rgbd_Gym import tm700_rgbd_gym
from bullet.tm700_rgb_Gym import tm700_rgb_gym

from bullet.tm700_possensor_Gym import tm700_possensor_gym
import numpy as np

from stable_baselines import DQN, DDPG
from baselines.helpers import savemodel
from datetime import date
import time
import baselines.parser as parser
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines import results_plotter
import matplotlib.pyplot as plt
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

args = parser.arg_parse()
set_global_seeds(args.random_seed)
start = time.time()
ENVIRONMENT = 'rgbd'
MODEL = 'DQN'
DISCRETE = True
DATE = date.today().strftime("%d-%m")
# DATE = str(time.time())
RENDERS = False
log_dir = ("./logdir_lr0001_%s_%s_%s/") % (MODEL, ENVIRONMENT, DATE)
time_steps = 100000
n_steps = 0
os.makedirs(log_dir, exist_ok=True)

################ MODEL AND GYM ENVIRONMENT

if ENVIRONMENT == 'rgbd':
  env = tm700_rgbd_gym(renders=RENDERS, isDiscrete=DISCRETE)
  env = Monitor(env, os.path.join(log_dir, 'monitor.csv'), allow_early_resets=True)


if ENVIRONMENT == 'rgb':
  env = tm700_rgb_gym(renders=RENDERS, isDiscrete=DISCRETE)
  env = Monitor(env, os.path.join(log_dir, 'monitor.csv'), allow_early_resets=True)

if ENVIRONMENT == 'possensor':
  env = tm700_possensor_gym(renders=RENDERS, isDiscrete=DISCRETE)
  env = Monitor(env, os.path.join(log_dir, 'monitor.csv'), allow_early_resets=True)

if MODEL == 'DQN':
  from stable_baselines.deepq.policies import LnCnnPolicy, MlpPolicy

  if ENVIRONMENT in ['rgbd', 'rgb', 'rgbdsparse']:
      model = DQN(LnCnnPolicy, env, verbose=1,
              tensorboard_log=(log_dir + "tensorboard_%s_%s_%s/") % (MODEL, ENVIRONMENT, DATE),
              gamma=0.99, learning_rate=0.0001, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
              train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=False, _init_setup_model=True,
              policy_kwargs=None, full_tensorboard_log=False)

  elif ENVIRONMENT in 'possensor':
      model = DQN(MlpPolicy, env, verbose=1, tensorboard_log=(log_dir + "tensorboard_%s_%s_%s/") % (MODEL, ENVIRONMENT, DATE) ,
              gamma=0.99, learning_rate=0.0005, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
               train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=False, _init_setup_model=True,
              policy_kwargs=None, full_tensorboard_log=False)

if MODEL == 'DDPG':
  from stable_baselines.ddpg.policies import LnCnnPolicy
  from stable_baselines.ddpg import AdaptiveParamNoiseSpec
  param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1)
  model = DDPG(LnCnnPolicy, env, verbose=1, random_exploration=0.1,tensorboard_log=(log_dir + "tensorboard_%s_%s_%s/") % (MODEL, ENVIRONMENT, DATE) )


################ CALLBACK FCTS

######################### PARAMETERS



def get_callback_vars(model, **kwargs):
    """
    Helps store variables for the callback functions
    :param model: (BaseRLModel)
    :param **kwargs: initial values of the callback variables
    """
    # save the called attribute in the model
    if not hasattr(model, "_callback_vars"):
        model._callback_vars = dict(**kwargs)
    else: # check all the kwargs are in the callback variables
        for (name, val) in kwargs.items():
            if name not in model._callback_vars:
                model._callback_vars[name] = val
    return model._callback_vars # return dict reference (mutable)



def auto_save_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if unintialized
    callback_vars = get_callback_vars(_locals["self"], n_steps=0, best_mean_reward=-np.inf)
    # skip every 20 steps
    if callback_vars["n_steps"] % 20 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > callback_vars["best_mean_reward"]:
                callback_vars["best_mean_reward"] = mean_reward
                # Example for saving best model
                print("Saving new best model at {} timesteps".format(x[-1]))
                _locals['self'].save(log_dir + 'best_model')
    callback_vars["n_steps"] += 1
    return True


def plotting_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    # get callback variables, with default values if unintialized
    callback_vars = get_callback_vars(_locals["self"], plot=None)

    # get the monitor's data
    x, y = ts2xy(load_results(log_dir), 'timesteps')
    if callback_vars["plot"] is None:  # make the plot
        plt.ion()
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        line, = ax.plot(x, y)
        callback_vars["plot"] = (line, ax, fig)
        plt.show()
    else:  # update and rescale the plot
        callback_vars["plot"][0].set_data(x, y)
        callback_vars["plot"][-2].relim()
        callback_vars["plot"][-2].set_xlim([_locals["total_timesteps"] * -0.02,
                                            _locals["total_timesteps"] * 1.02])
        callback_vars["plot"][-2].autoscale_view(True, True, True)
        callback_vars["plot"][-1].canvas.draw()


def compose_callback(*callback_funcs): # takes a list of functions, and returns the composed function.
    def _callback(_locals, _globals):
        continue_training = True
        for cb_func in callback_funcs:
            if cb_func(_locals, _globals) is False: # as a callback can return None for legacy reasons.
                continue_training = False
        return continue_training
    return _callback

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy training performance
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True


################ TRAINING

model.learn(total_timesteps=time_steps, callback=auto_save_callback, seed=args.random_seed)

# print('save model')
# savemodel(model, MODEL, ENVIRONMENT, DATE)

results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "RGBD Observation")
plt.show()
print('total time', time.time()-start)


