#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def get_reward(error):
    """Uses current pose of sim to return reward."""
    reward = 1/(error+1)
    #reward = 1 - error
    return reward


max_err = (abs(np.array([0.,0.,10.]) - np.array([10.,10.,0.]))).sum()

errors = np.linspace(0,max_err,1000)
count = []
reward = []

#errors = [0,1,2,3,4,5,6,7,8,9]
#reward = [9,8.5,6.5,4.5,3.5,3.2,3,2,1,0]
for i in errors:
    reward.append(get_reward(i))

print("error:",list(errors))
print("reward",reward)   
plt.plot(errors, reward, label='reward')
plt.legend()
plt.xlabel('errors')
plt.ylabel('rewards')
plt.title('Rewards curve')
