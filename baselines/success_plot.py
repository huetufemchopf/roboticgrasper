import numpy as np
from stable_baselines.results_plotter import ts2xy,load_results
import matplotlib.pyplot as plt
from stable_baselines import results_plotter

################## LOAD DATA

possensor = './logdir_DDPG_possensorbothgrippers_10-02'
rgbd ='./logdir_DQN_rgbd_07-02/'
rgb = './logdir_DQN_rgb_07-02/'
rgbdsparse = './logdir_DQN_rgbdsparse_09-02/'

lr005 = './logdir_lr005_DQN_rgbd_08-02/'
lr00005 = './logdir_lr00005_DQN_rgbd_08-02/'
lr0001 = './logdir_lr0001_DQN_rgbd_10-02/'

g79 = './logdir_g79_DQN_rgbd_08-02/'
g89 = './logdir_g89_DQN_rgbd_08-02/'
normalized = './logdir_normalized_DQN_rgbd_08-02'
multi = './logdir_DQN_multi_10-02'
multi2 = 'logdir_lr005_DQN_multi_10-02'
segmentation = './logdir_segmentation_DQN_rgbd_11-02'
def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)

    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    # plt.show()


def plot_successrate(log_folder, model, title='Success Rate'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    n=0
    episodes = []
    for i in range(len(x)):
        episodes.append(n)
        n+=1

    successrate = []
    nrofgrasps = 0
    j = 0
    for i in y:
        if i > 900:
            nrofgrasps +=1
        j +=1
        successrate.append(nrofgrasps/j)

    # Truncate x

    fig = plt.figure(title)
    plt.plot(episodes, successrate, label=model)
    # plt.xlim((0, 8000))
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title(title)


plot_successrate(rgbd, 'RGBD')
plot_successrate(rgb,'RGB' )
plot_successrate(rgbdsparse,'RGBD Sparse Reward' )
plot_successrate(segmentation,'RGB plus Segmetation' )
plot_successrate(possensor, 'Possensor')

plt.legend()
plt.show()

# results_plotter.plot_results([rgbd], 10000, results_plotter.X_TIMESTEPS, "RGB Observation")
# results_plotter.plot_results([rgb], 10000, results_plotter.X_TIMESTEPS, "RGB Observation")
# results_plotter.plot_results([possensor], 10000, results_plotter.X_TIMESTEPS, "RGB Observation")
# results_plotter.plot_results([rgbdsparse], 10000, results_plotter.X_TIMESTEPS, "RGB Observation")
# plot_results(rgbd)
# plot_results(rgb)
# plt.show()


# plot_successrate(rgbd, 'RGBD, Learning Rate = 5e-4')
# plot_successrate(lr005, 'RGBD, Learning Rate = 5e-3')
# plot_successrate(lr0001, 'RGBD, Learning Rate = 1e-3')
# plot_successrate(lr00005, 'RGBD, Learning Rate = 5e-5')
# plt.legend()
# plt.show()

# plot_successrate(rgbd, 'RGBD, Discount Factor = 0.99')
# plot_successrate(g79, 'RGBD, Discount Factor = 0.79')
# plot_successrate(g89, 'RGBD, Discount Factor = 0.89')
# plt.legend()
# plt.show()


# plot_successrate(rgbd, 'RGBD')
# plot_successrate(normalized, 'RGBD, Normalized Depth Camera')
# plt.legend()
# plt.show()
#
# plot_successrate(possensor, 'Position Sensor')
# plt.legend()
# # plt.show()
# plot_results(rgbd)
# plot_results(rgb)
# plot_results(rgbdsparse)
# plt.xlim(0,100000)
#
# plt.show()


plot_results(possensor)
plt.legend()
plt.show()