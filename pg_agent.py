from pg import PolicyGradient
import numpy as np
import time
from time import gmtime, strftime

import gym_remote.exceptions as gre
import gym_remote.client as grc

def main():
    env = grc.RemoteEnv('tmp/sock')

    # Policy gradient has high variance, seed for reproducability
    env.seed(1)

    RENDER_ENV = False
    rewards = []
    INITIAL_EPSILON = 0.7
    EPSILON_GREEDY_INCREMENT = 0.01

    # Load checkpoint
    load_version = "2018-06-05 18:24:13"
    timestamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    load_path = "output/model2/{}/SonicTheHedgehog.ckpt".format(load_version)

    PG = PolicyGradient(
        n_x = [112,112,3], #env.observation_space.shape,
        n_y = env.action_space.n,
        learning_rate=0.02,
        reward_decay=0.99,
        load_path=load_path,
        epsilon_max=0.98,
        epsilon_greedy_increment=EPSILON_GREEDY_INCREMENT,
        initial_epsilon = INITIAL_EPSILON
    )

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

        # Save new observation
        observation = observation_

        if done:
            episode_rewards_sum = sum(PG.episode_rewards)
            rewards.append(episode_rewards_sum)
            max_reward_so_far = np.amax(rewards)

            if episode_rewards_sum == 0.0:
                print("-----------------------------------")
                print("Backtrack epsilon for more exploration...")
                PG.epsilon = max(PG.epsilon - EPSILON_GREEDY_INCREMENT, INITIAL_EPSILON)

            print("==========================================")
            print("Epsilon: ", PG.epsilon)
            print("Seconds: ", elapsed_sec)
            print("Reward: ", episode_rewards_sum)
            print("Max reward so far: ", max_reward_so_far)

            # 5. Train neural network
            tic = time.clock()
            discounted_episode_rewards_norm = PG.learn()
            toc = time.clock()
            elapsed_sec = toc - tic
            print("Train Seconds: ", elapsed_sec)

            observation = env.reset()


if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as e:
        print('exception', e)
