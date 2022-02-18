# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense,Input, Flatten
from keras.optimizers import Adam
from keras import backend as K
import gym
import minerl
import tensorflow as tf
from keras.models import Model
from rl.agents import DDPGAgent

EPISODES = 4
if __name__ == "__main__":
    env = gym.make('MineRLTreechop-v0')
    obs, _ = env.reset()
    state_size = (12288,)
    action_size = 10
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-ddqn.h5")
    done = False
    batch_size = 32
    for e in range(EPISODES):
        state, _ = env.reset()
        state = obs['pov'].flatten()
        state.shape
        state = np.reshape(state, [1, 12288])
        for time in range(500):
            # env.render()
            action = env.action_space.noop()
            count = 0
            action_predict = agent.act(state)
            for key, i in action.items():
                if key != 'camera':
                    action[key] = action_predict[count]
                else:
                    action[key] = [action_predict[-1],action_predict[-2]]
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -1
            next_state = next_state['pov'].flatten()
            next_state = np.reshape(next_state, [1, 12288])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-ddqn.h5")