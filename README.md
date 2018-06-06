# Retro Contest Open AI
### https://contest.openai.com/

# Reinforcement Learning with Policy Gradient
### Uses 2 convolution layers and 1 fully connected layer to control Sonic

Video of agent playing Sonic! -> https://contest.openai.com/users/911

[![sonic](https://user-images.githubusercontent.com/1076706/41054460-4a55987e-6973-11e8-8e2b-5d48045a757d.png)](https://contest.openai.com/users/911)

1) To run this code first `retro-contest` and follow the instructions on the contest page: https://contest.openai.com/details

2) `pg_agent_train.py` trains the agent over defined episodes. You can change the SonicTheHedgehog level and act in that file to train on different levels. `pg.py` contains the model used for training.

3) `pg_agent.py` was used for submission.

4) With only half day of training on CPU, this agent was not able to generalize well enough but still achieved a decent score

5) Eventually ran a PPO baseline and was able to rank 140 out of 229 participants: https://contest.openai.com/leaderboard
