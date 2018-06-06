from retro_contest.local import make


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        action[7] = 1
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
