import sys
import numpy as np
import pandas as pd
import time
import datetime
from agents.DDPGAgent import DDPG
from task import Task
from collections import deque
import math
import matplotlib.pyplot as plt

num_episodes = 100
target_pos = np.array([0., 0., 10.])
task = Task(target_pos=target_pos)

# performace vars 
runtime = 5.                                     # time limit of the episode
init_pose = np.array([0., 0., 10., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])


## setting up noise process values
#mu = np.random.permutation([0, 0, 0, 0])
#theta = np.random.permutation([0.1, 0.12, 0.15, 0.17])
#sigma = np.random.permutation([0.2, 0.2, 0.2, 0.2])
#
## setting up algorithm parameters
#gamma = np.random.permutation([0.8, 0.9, 0.99, 1.1])  # discount factor
#tau = np.random.permutation([0.01, 0.01, 0.01, 0.01])  # for soft update of target parameters

# setting up noise process values
mu = np.array([0])
theta = np.array([0.15])
sigma = np.array([0.2])

# setting up algorithm parameters
gamma = np.array([0.99])  # discount factor
tau = np.array([0.01])  # for soft update of target parameters

# training batches
batches = np.array([64])

# log path
path = 'reward_logs/'
        
def perform(save = False, learn=True):
    
    # initialize rewards
    best_avg_reward = -math.inf
    samp_rewards = deque(maxlen=30)
    
    # setting up log parameters
    ts = time.time()
    logtime = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    
    # logs
    reward_log_file = 'reward_log_{}.csv'.format(logtime)
    perform_log_file = 'perform_log_{}.csv'.format(logtime)
    
    # log columns
    perform_cols = ['time', 'x', 'y', 'z', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 
              'rotor_speed4']
    reward_cols = ['Episode', 'Reward'] 
    
    # dataframes init
    reward_log = pd.DataFrame(index=None, columns=reward_cols) 
    perform_log = pd.DataFrame(index=None, columns=perform_cols)
    
    # performace / training loop
    for i_episode in range(1, num_episodes+1):
        state = agent.reset_episode() # start a new episode
        samp_reward = 0
        while True:
            action = agent.act(state, learn)
            next_state, reward, done = task.step(action)
            if learn:
                agent.step(action, reward, next_state, done, save)
            else:
                perform_log = perform_log.append(pd.DataFrame(data=[[task.sim.time,
                                                                 task.sim.pose[0],
                                                                 task.sim.pose[1],
                                                                 task.sim.pose[2],
                                                                 task.sim.v[0],
                                                                 task.sim.v[1],
                                                                 task.sim.v[2],
                                                                 task.sim.angular_v[0],
                                                                 task.sim.angular_v[1],
                                                                 task.sim.angular_v[2],
                                                                 action[0],
                                                                 action[1],
                                                                 action[2],
                                                                 action[3]]],
                                                      columns=perform_cols), ignore_index=True)
            samp_reward += reward
            state = next_state
            if done:
                samp_rewards.append(samp_reward)
                break
        # get average reward 
        avg_reward = np.mean(samp_rewards)
        # update best average reward
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save = True                
        elif avg_reward < best_avg_reward:
            save = False
        #monitor progress
        if learn:
            reward_log = reward_log.append(pd.DataFrame(data=[[i_episode, avg_reward]],
                                                      columns=reward_cols), ignore_index=True)
        print("\r{}: Episode {}/{} Avg reward {} || Best avg reward {}".format(logtime, i_episode,
              num_episodes, round(avg_reward,3), round(best_avg_reward,3)), end="")
        sys.stdout.flush()
    # convert data frame to csv - output results
    if learn:
        reward_log.to_csv(path + reward_log_file,  index=False)
    else:
        perform_log.to_csv(path + perform_log_file,  index=False)
    return reward_log_file, perform_log_file, best_avg_reward

# plot functions
def plot_rewards(path, fl, m, th, s, g, ta, b):
    data = pd.read_csv(path + fl)
    x, y = np.array(data['Episode']), np.array(data['Reward'])
    plt.plot(x, y, label='m={} th={} s={} g={} t={} b={}'.format(m, th, s, g, ta, b))
    plt.legend()
    plt.title('mu - theta - sigma')
    plt.title('gamma - ', loc='left')
    plt.title(' - tau - batch', loc='right')
    
def plot_q_pos(path, fl):
    data = pd.read_csv(path + fl)
    t, x, y, z = np.array(data['time']), np.array(data['x']), np.array(data['y']), np.array(data['z'])
    plt.plot(t, x, label='x')
    plt.plot(t, y, label='y')
    plt.plot(t, z, label='z')
    plt.legend()
    plt.title('Position of quadcopter')
    
def plot_q_v(path, fl):
    data = pd.read_csv(path + fl)
    t, xv, yv, zv = np.array(data['time']), np.array(data['x_velocity']), np.array(data['y_velocity']), np.array(data['z_velocity'])
    plt.plot(t, xv, label='x-velo')
    plt.plot(t, yv, label='y-velo')
    plt.plot(t, zv, label='z-velo')
    plt.legend()
    plt.title('Velocity of quadcopter')
    
def plot_q_va(path, fl):
    data = pd.read_csv(path + fl)
    t, xav, yav, zav = np.array(data['time']), np.array(data['phi_velocity']), np.array(data['theta_velocity']), np.array(data['psi_velocity'])
    plt.plot(t, xav, label='phi-avelo')
    plt.plot(t, yav, label='theta-avelo')
    plt.plot(t, zav, label='psi-avelo')
    plt.legend()
    plt.title('Angular velocity of quadcopter')

def plot_q_r(path, fl):
    data = pd.read_csv(path + fl)
    t, r1, r2, r3, r4 = np.array(data['time']), np.array(data['rotor_speed1']), np.array(data['rotor_speed2']), np.array(data['rotor_speed3']), np.array(data['rotor_speed4'])
    plt.plot(t, r1, label='r1_rps')
    plt.plot(t, r2, label='r2_rps')
    plt.plot(t, r3, label='r3_rps')
    plt.plot(t, r4, label='r4_rps')
    plt.legend()
    plt.title('Rotors speed')
    
# training agent with predefined parameters to get best results
for i in range(1):
    for m, th, s, g, ta, b in zip(mu, theta, sigma, gamma, tau, batches):    
        print('\n \nTry with parameters: mu={}; theta={}; sigma={}, gamma={}, tau={}, batch={}'.format(m, th, s, g, ta, b))
        agent = DDPG(task, m, th, s, g, ta, b)
        result = perform(learn=True)
        # plotting
        plot_rewards(path, result[0], m, th, s, g, ta, b)
        m, th, s, g, ta, b = m, th, s, g, ta, b

## demoing model within 10 episodes
#print("\n \nDemo simulation: ")
#num_episodes = 10
#m, th, s, g, ta = 0, 0.15, 0.2, 0.99, 0.01
#runtime = 10
#task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos)
#agent = DDPG(task, m, th, s, g, ta)
#result = perform(learn=False)
#print('Parameters: m={} th={} s={} g={} t={}'.format(m, th, s, g, ta))
#    
## demoing performance   
#print("\n \nDemo simulation: ")
#num_episodes = 1
#runtime = 5
#task = Task(init_pose, init_velocities, init_angle_velocities, runtime, target_pos=target_pos)
#agent = DDPG(task, m, th, s, g, ta)
#print('Parameters: m={} th={} s={} g={} t={}'.format(m, th, s, g, ta))
#result = perform(learn=False)
#plot_q_pos(path, result[1])
#plot_q_v(path, result[1])
#plot_q_va(path, result[1])
#plot_q_r(path, result[1])
