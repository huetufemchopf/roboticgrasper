#add parent dir to find package. Only needed for source code build, pip install doesn't need it.
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
print(parentdir)

os.sys.path.insert(0, parentdir)
import numpy as np

from bullet.tm700_possensor_Gym import tm700_possensor_gym

from stable_baselines import DQN, DDPG, HER
# from stable_baselines_test.deepq.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy
import time

from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy

######################### PARAMETERS

log_dir = './log_dir/'
os.makedirs(log_dir, exist_ok=True)


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




def train_cam():
    param_noise = None

    env1 = tm700CamGymEnv(renders=True, isDiscrete=False)
    model = DQN(MlpPolicy, env1, verbose=1,
                 tensorboard_log="./tensorboard_dqn_tm700/", gamma=0.99, learning_rate=0.0005,
                buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
              exploration_initial_eps=1.0, train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
              target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
              param_noise=param_noise, n_cpu_tf_sess=None, _init_setup_model=True,
              policy_kwargs=None, seed=None)
    # model = DQN(MlpPolicy, env1, verbose=1, exploration_fraction=0.3)

    # = deepq.models.mlp([64])
    start = time.time()
    model.learn(total_timesteps=1000000)
    # max_timesteps=10000000,
    # exploration_fraction=0.1,
    # exploration_final_eps=0.02,
    # print_freq=10,
    # callback=callback, network='mlp')
    print("Saving model")
    model.save("tm_test_model_randomblocks2dqn.pkl")

    print('total time', time.time() - start)

def train_base():
  param_noise = None

  env1 = tm700_possensor_gym(renders=False, isDiscrete=False)
  model = DDPG(MlpPolicy, env1, verbose=1, param_noise=param_noise, random_exploration=0.1, tensorboard_log="./tensorboard_ddpg_tm700/")
  # model = DQN(MlpPolicy, env1, verbose=1,
  #             tensorboard_log="./tensorboard_dqn_tm700/", gamma=0.9, learning_rate=0.0005,
  #             buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02,
  #              train_freq=1, batch_size=32, double_q=True, learning_starts=1000,
  #             target_network_update_freq=500, prioritized_replay=True, prioritized_replay_alpha=0.6,
  #             prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None, prioritized_replay_eps=1e-06,
  #             param_noise=param_noise,_init_setup_model=True,
  #             policy_kwargs=None)
  # model = DQN(MlpPolicy, env1, verbose=1, exploration_fraction=0.3)

  start = time.time()
  env = Monitor(env1, log_dir, allow_early_resets=True)

  model.learn(total_timesteps=1000000, callback=auto_save_callback)

  print("Saving model")
  model.save("trainedmodel_randomblocks_0402.pkl")

  print('total time', time.time()-start)

def heralgorithm():

    goal_selection_strategy = 'future'  # equivalent to GoalSelectionStrategy.FUTURE

    # Wrap the model
    model = HER('MlpPolicy', env1, DDPG, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                verbose=1)
    # Train the model
    model.learn(1000)

    model.save("./her_bit_env")



train_base()