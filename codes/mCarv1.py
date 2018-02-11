import gym
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
from collections import deque
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

class Replay_Buffer:
    def __init__(self, size = 50000):
        self.buffer = deque(maxlen=size)

    def add(self, item):
        self.buffer.append(item)

    def sample(self, sample_size):
        choices = np.random.choice(len(self.buffer),size=sample_size,replace=False)
        return [self.buffer[x] for x in choices]

class Agent:
    def __init__(self, lr, h_size, s_size=2, a_size=3):
        self.model = self.makeModel(lr, h_size, s_size, a_size)
        self.target_model = self.makeModel(lr, h_size, s_size, a_size)

    def makeModel(self, lr, h_size, s_size, a_size):
        model = Sequential()

        model.add(Dense(h_size, input_dim=2,  init='lecun_uniform',))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(h_size,  init='lecun_uniform',))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(h_size,  init='lecun_uniform',))
        model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(a_size, activation='linear'))
        adam = optimizers.Adam(lr=lr, decay=1e-6)
        model.compile(loss='mse', optimizer=adam)
        model.summary()
        return model

    def updateTarget(self):
        self.target_model.set_weights(self.model.get_weights())

    def getAction(self, state):
        s = np.array([state])
        return np.argmax(self.model.predict(s)[0])

def upThatHill(lr=0.0005, h_size=256, buff_sz=85000, batch_sz=256, batch_freq=1, update_target_freq=2, gamma=0.99, random_eps=25, train_eps=5000, anneal_eps=7500):
    a_size = 3 # Number of actions
    s_size = 2 # Number of state observations
    dqn = Agent(lr, h_size, s_size, a_size) # Initialize the agents
    myBuffer = Replay_Buffer(buff_sz) # Initialize action replay
    #Set epsilon decay
    e = 1
    min_e = 0
    decay = (1 - min_e)/anneal_eps
    all_rewards = []
    avg_rewards = deque(maxlen=100) # Length of win condition
    max_reward = -200 # Starts at the lowest possible reward - for tracking new maximum rewards during training

    maxNegPosition = 9999999
    maxPosPosition = -9999999

    for i in range(train_eps):
        # Reset the environment for new episode
        actions = []
        s = env.reset()
        d = False
        ep_reward = 0

        # While this episode is not done
        while not d:
            env.render()
            # Choose action - either randomly (exploration) or by highest value (exploitation)
            if i < random_eps or np.random.rand(1) < e:
                a = np.random.randint(0, a_size)
            else:
                a = dqn.getAction(s)

            # Take action and save experience to replay buffer
            actions.append(a)
            s1,r,d,_ = env.step(a)

            tempReward = r

            # position and velocity...
            newPosition = s1[0]
            velocity = s1[1]

            positionDiff = newPosition + 0.5
            tempReward = abs(positionDiff) * abs(velocity)


            # if s1[0] < maxNegPosition:
            #     maxNegPosition = s1[0]
            #     tempReward = 1
            #
            # if s1[0] > maxPosPosition:
            #     maxPosPosition = s1[0]
            #     tempReward = 1

            myBuffer.add((s,a,tempReward,s1,d))

            # Update the target model if necessary
            if i % update_target_freq == 0:
                dqn.updateTarget()

            if i > random_eps:
                # Perform learning
                batch = myBuffer.sample(batch_sz)
                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []
                for vals in batch:
                    states.append(vals[0])
                    actions.append(vals[1])
                    rewards.append(vals[2])
                    next_states.append(vals[3])
                    dones.append(vals[4])

                y = np.zeros((batch_sz, a_size))
                q = dqn.model.predict(np.array(states))
                q1 = dqn.target_model.predict(np.array(next_states))
                x = np.array(states)

                for k in range(batch_sz):
                    target = q[k]
                    target_action = rewards[k]
                    if not dones[k]:
                        target_action += (gamma * max(q1[k]))
                    target[actions[k]] = target_action
                    y[k, :] = target
                dqn.model.fit(x, y, batch_size=batch_sz, epochs=1, verbose=False)

            # Add reward to episode reward and make current state into previous state
            ep_reward += r
            s = s1

        # Decay exploration factor
        if i > random_eps and e > min_e:
            e -= decay

        # Add rewards to running average and all rewards
        avg_rewards.append(ep_reward)
        all_rewards.append(ep_reward)

        # Report when a new best reward is found
        if ep_reward > max_reward:
            max_reward = ep_reward
            print("New max!", max_reward)

        # Go to test phase if agent has learned enough
        if sum(avg_rewards)/len(avg_rewards) > -105:
            break

        #Periodically report on the model.
        if i % 200 == 0:
            print("Episode", i)
            print("Avg reward", sum(avg_rewards)/len(avg_rewards))

    print("Avg reward", sum(avg_rewards)/len(avg_rewards))
    print("Done training")
    print("Total episodes: ", i)

    test_rewards = []
    test_actions = []
    test_states = []
    for test in range(100): # Test trained model
        actions = []
        states = []
        s = env.reset()
        d = False
        ep_reward = 0
        while not d:
            a = dqn.getAction(s)
            actions.append(a)
            states.append(s)
            s1,r,d,_ = env.step(a)
            ep_reward += r
            s = s1

        test_rewards.append(ep_reward)
        test_actions.append(actions)
        test_states.append(states)
    print("Test complete")
    print("Test result:", sum(test_rewards)/len(test_rewards))
    return all_rewards, avg_rewards, test_rewards, test_actions, test_states






'''..................................main Method ..............................'''
env = gym.make('MountainCar-v0')

env.reset()
random_rewards = []
random_actions = []
random_position = []
random_velocity = []

total_reward = 0
done = False
for _ in range(200): #This is the maximum length of an episode for this environment
    if not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        position = state[0]
        velocity = state[1]
    else:  #If the episode ends early append dummy values - this will make the final comparison graphs cleaner
        reward = 0
        action = 3
        position = random_position[-1]
        velocity = random_velocity[-1]

    total_reward += reward
    random_rewards.append(total_reward)
    random_actions.append(action)
    random_position.append(position)
    random_velocity.append(velocity)
env.reset()
baseline = upThatHill(h_size=30)
