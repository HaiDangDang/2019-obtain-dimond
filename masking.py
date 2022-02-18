# Derived from keras-rl
import numpy as np
import sys
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate,Convolution2D, Permute,Concatenate,Reshape
from keras.optimizers import Adam
import minerl
import importlib
import numpy as np
import keras.backend as K
from rl.agents import  DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess,RandomProcess
import gym
from rl.   policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from PIL import Image
import cv2
from keras.optimizers import RMSprop
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_images
from masking_objects.ballon_sample_mask import BalloonDataset,BalloonConfig,color_splash
importlib.reload(modellib)
INPUT_SHAPE = (64, 64)
WINDOW_LENGTH = 1
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
from PIL import Image
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = BalloonConfig()

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
env = gym.make('MineRLTreechop-v0')
delta_yaw = [-90, -60, -45,- 30, -25, -15, -10, 0, 10, 15, 25, 30, 45, 69, 90]
# model_woood = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# # Load weights trained on MS-COCO
# model_woood.load_weights('mask_rcnn_balloon_0030.h5', by_name=True)


class TreechopProcessor(Processor):
    sum_pitch = 0
    curremt_yaw = 0
    def process_step(self, observation, reward, done, info):
        if done == True:
            self.sum_pitch = 0
            self.curremt_yaw = 0
            # results = model_woood.detect([observation['pov']])
        # r = results[0]
        # sum_tree = 0
        # for i in r['masks']:
        #     sum_tree += (i == True).sum()
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        #print(sum_tree*0.0001)
        return observation, reward, done, info

    def process_observation(self, observation):
        if len(observation) > 1:
            observation = observation[0]['pov']
        else:
            observation = observation['pov']
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = observation.flatten()
        #observation = np.reshape(observation, [1, 12288])
        return observation  # saves storage in experience memory

    def process_state_batch(self, batch):
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return reward
    def process_action(self, action):
        v1 = action[0] * 9
        v1 = np.clip(v1,0,9)
        action_tmp = [0,0,0,0,0,0]
        if 0 <= v1 <= 1:
            action_tmp = [1,0,0,0,0,0]
        elif 1 <= v1 <= 2:
            action_tmp = [0,1,0,0,0,0]
        elif 2 <= v1 <= 3:
            action_tmp = [0,0,1,0,0,0]
        elif 3 <= v1 <= 4:
            action_tmp = [0,0,0,1,0,0]
        elif 4 <= v1 <= 5:
            action_tmp = [0,0,0,0,1,0]
        elif 5 <= v1 <= 6:
            action_tmp = [0,1,0,0,0,1]
        elif 6 <= v1 <= 7:
            action_tmp = [0,0,1,0,0,1]
        elif 7 <= v1 <= 8:
            action_tmp = [0,0,0,1,0,1]
        elif 8 <= v1 <= 9:
            action_tmp = [0,0,0,0,1,1]
        action_1 = env.action_space.noop()
        action_1['attack'] = action_tmp[0]
        action_1['back'] = action_tmp[1]
        action_1['forward'] = action_tmp[2]
        action_1['left'] = action_tmp[3]
        action_1['right'] = action_tmp[4]
        action_1['jump'] = action_tmp[5]

        v2 = action[1] * 5
        v2 = np.clip(v2, 0, 5)
        pitch = 0
        angle =0
        if 0 <= v1 <= 1:
            pitch = 0
        elif 1 <= v2 <= 4:
            angle = (action[1]*5 -2.5) * 60
            angle = np.clip(angle, -90, 90)
            pitch = angle - self.sum_pitch
            self.sum_pitch = np.int(np.rint(angle))
        elif 4 <= v2 <= 5:
            pitch = np.invert(self.sum_pitch) + 1
            self.sum_pitch = 0
        v3 = action[2] * 2
        v3 = np.clip(v3, 0, 2)
        yaw = 0
        angle = 0
        if 0 <= v3 <= 1:
            yaw = 0
        elif 1 <= v3 <= 2:
            angle = (v3 - 1) * 360
            angle = np.clip(angle, 0, 360)
            if angle == 360:
                angle = 0
            yaw = angle - self.curremt_yaw
            yaw2 = (angle - self.curremt_yaw) - 360
            yaw = yaw2 if np.abs(yaw2) < np.abs(yaw) else yaw
            self.curremt_yaw = angle

        action_1['camera'] = [pitch,
                              yaw]
        print(f'yaw: {yaw}, angle: {angle}, sum: {self.curremt_yaw}, action: {action[2]}, v3:{v3}')
        return action_1


action_size = 3
nallsteps = 20000000
nb_actions = 3
model = Sequential()
model.add(Reshape((64, 64, 1), input_shape=(1,) + (4096,)))
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('sigmoid'))
print(model.summary())


action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) +(4096,) , name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

processor = TreechopProcessor()
# Set up the agent for training
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=10000)

memory = SequentialMemory(limit=1000000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.01, mu=0., sigma=.1)

agent = DDPGAgent(nb_actions=nb_actions, processor=processor, actor=model, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=10000, nb_steps_warmup_actor=10000,
                   gamma=.99, target_model_update=1e-3,random_process=random_process)

agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
agent.fit(env, nb_steps=nallsteps, verbose=1, nb_max_episode_steps=18000, log_interval=10000)



