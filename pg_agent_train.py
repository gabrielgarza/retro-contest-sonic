from retro_contest.local import make
from pg import PolicyGradient
# import matplotlib.pyplot as plt
import numpy as np
import time
from time import gmtime, strftime

import gym_remote.exceptions as gre
import gym_remote.client as grc
# env = grc.RemoteEnv('tmp/sock')

env = make(game='SonicTheHedgehog-Genesis', state='StarLightZone.Act1')

# SpringYardZone.Act3
# SpringYardZone.Act2
# GreenHillZone.Act3
# GreenHillZone.Act1
# StarLightZone.Act2
# StarLightZone.Act1
# MarbleZone.Act2
# MarbleZone.Act1
# MarbleZone.Act3
# ScrapBrainZone.Act2
# LabyrinthZone.Act2
# LabyrinthZone.Act1
# LabyrinthZone.Act3

# SpringYardZone.Act1
# GreenHillZone.Act2
# StarLightZone.Act3
# ScrapBrainZone.Act1

# Policy gradient has high variance, seed for reproducability
env.seed(1)

# print("env.action_space", env.action_space)
# print("env.action_space.n", env.action_space.n)
# print("env.observation_space", env.observation_space)
# print("env.observation_space.high", env.observation_space.high)
# print("env.observation_space.low", env.observation_space.low)

# How to show an image
# fig, ax = plt.subplots(ncols = 1)
# ax.imshow(observation)
# plt.show()

# Model1
# Maxpool layers and 4 channels for both layers

# Model2
# No maxpools and channels are 8 and 16

RENDER_ENV = False
EPISODES = 1000
rewards = []
RENDER_REWARD_MIN = 7000
MIN_REWARD_TO_LEARN = -10
EARLY_TERM_REWARD = 100
EARLY_TERM_EXPLORATION_EPSILON = 0.8
EARLY_TERM_OBS = 5000
INITIAL_EPSILON = 0.7
EPSILON_GREEDY_INCREMENT = 0.01

if __name__ == "__main__":

    # Load checkpoint
    load_version = "2018-06-05 16:31:58"
    timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    load_path = "output/model2/{}/SonicTheHedgehog.ckpt".format(load_version)
    save_path = "output/model2/{}/SonicTheHedgehog.ckpt".format(timestamp)

    PG = PolicyGradient(
        n_x = [112,112,3], #env.observation_space.shape,
        n_y = env.action_space.n,
        learning_rate=0.02,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path,
        epsilon_max=0.98,
        epsilon_greedy_increment=EPSILON_GREEDY_INCREMENT,
        initial_epsilon = INITIAL_EPSILON
    )


    for episode in range(EPISODES):

        observation = env.reset()
        # print("obs", observation)
        episode_reward = 0

        tic = time.clock()

        while True:
            if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            observation = observation[:,96:,:] # make square, keep right sight of image
            observation = observation[::2,::2,:] # downsample to [112,112,3]
            observation = observation / 255 # normalize
            action = PG.choose_action(observation)
            
            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 4. Store transition for training
            PG.store_transition(observation, action, reward)

            episode_rewards_sum = sum(PG.episode_rewards)

            toc = time.clock()
            elapsed_sec = toc - tic

            # if len(PG.episode_observations) >= EARLY_TERM_OBS and episode_rewards_sum < EARLY_TERM_REWARD:
            #     print("-----------------------------------")
            #     print("Early termination - low rewards...")
            #     done = True

            if PG.epsilon < EARLY_TERM_EXPLORATION_EPSILON and len(PG.episode_observations) >= EARLY_TERM_OBS:
                print("-----------------------------------")
                print("Early termination - exploration...")
                done = True


            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                if episode_rewards_sum == 0.0:
                    print("-----------------------------------")
                    print("Backtrack epsilon for more exploration...")
                    PG.epsilon = max(PG.epsilon - EPSILON_GREEDY_INCREMENT, INITIAL_EPSILON)

                print("==========================================")
                print("Episode: ", episode)
                print("Epsilon: ", PG.epsilon)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)

                # 5. Train neural network
                tic = time.clock()
                if episode_rewards_sum > MIN_REWARD_TO_LEARN:
                    discounted_episode_rewards_norm = PG.learn()
                toc = time.clock()
                elapsed_sec = toc - tic
                print("Train Seconds: ", elapsed_sec)

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_
