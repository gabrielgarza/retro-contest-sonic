"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
# from tensorflow.python.framework import ops

class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate=0.01,
        reward_decay=0.95,
        load_path=None,
        save_path=None,
        epsilon_max=0.95,
        epsilon_greedy_increment=0.001,
        initial_epsilon = 0.8,
        batch_size = 400
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = epsilon_max
        self.epsilon_greedy_increment = epsilon_greedy_increment
        self.batch_size = batch_size

        if epsilon_greedy_increment is not None:
            self.epsilon = initial_epsilon
        else:
            self.epsilon = self.epsilon_max

        self.save_path = None
        if save_path is not None:
            self.save_path = save_path

        self.episode_observations, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

        # Restore model
        if load_path is not None:
            self.load_path = load_path
            self.saver.restore(self.sess, self.load_path)

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: observation
                a: action taken
                r: reward after action
        """
        self.episode_observations.append(s)
        self.episode_rewards.append(r)
        self.episode_actions.append(a)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """

        if np.random.uniform() < self.epsilon:
            # print("Predict Action")

            # Reshape observation to (num_features, 1)
            observation = observation[np.newaxis, :]

            # Run forward propagation to get softmax probabilities
            prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

            # Select action using a biased sample
            # this will return the index of the action we've sampled
            action_index = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())

            actions = np.zeros(12)
            actions[action_index] = 1
        else:
            # print("Take Random Action")
            actions = np.zeros(12)
            random_index = np.random.randint(0,12)
            actions[random_index] = 1

        # print("prob_weights", prob_weights)
        # print("action_index", action_index)
        # print("actions", actions)

        return actions

    def learn(self):
        index_range = len(self.episode_observations)
        sample_index = np.random.choice(index_range, size=self.batch_size)
        batch_memory_s = np.array(self.episode_observations)[sample_index, ...]
        batch_memory_a = np.array(self.episode_actions)[sample_index, ...]
        batch_memory_r = np.array(self.episode_rewards)[sample_index, ...]

        # Discount and normalize episode reward
        discounted_episode_rewards_norm = self.discount_and_norm_rewards(batch_memory_r)

        # Train on episode
        print("Training on batch_memory_s shape: ", batch_memory_s.shape)


        self.sess.run(self.train_op, feed_dict={
             self.X: batch_memory_s,
             self.Y: batch_memory_a,
             self.discounted_episode_rewards_norm: discounted_episode_rewards_norm,
        })

        # Reset the episode data
        self.episode_observations, self.episode_actions, self.episode_rewards  = [], [], []

        # Save checkpoint
        if self.save_path is not None:
            save_path = self.saver.save(self.sess, self.save_path)
            print("Model saved in file: %s" % save_path)

        # Increase epsilon to make it more likely over time to get actions from predictions instead of from random sample
        if self.epsilon_greedy_increment is not None:
            self.epsilon = np.round(min(self.epsilon_max, self.epsilon + self.epsilon_greedy_increment),3)

        return discounted_episode_rewards_norm

    def discount_and_norm_rewards(self, batch_rewards):
        # discount episode rewards
        discounted_episode_rewards = np.zeros_like(batch_rewards)
        running_add = 0
        for t in reversed(range(0, len(batch_rewards))):
            running_add = running_add * self.gamma + batch_rewards[t]
            discounted_episode_rewards[t] = running_add

        # normalize episode rewards
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        n_H0 = self.n_x[0]
        n_W0 = self.n_x[1]
        n_C0 = self.n_x[2]
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.n_y), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, ], name="actions_value")

        # Initialize parameters
        input_channels = 3
        filter_layer_1 = 4
        channels_layer_1 = 8
        filter_layer_2 = 2
        channels_layer_2 = 16
        units_output_layer = self.n_y
        with tf.name_scope('parameters'):
            W1 = tf.get_variable("W1", [filter_layer_1, filter_layer_1, input_channels, channels_layer_1], initializer = tf.contrib.layers.xavier_initializer(seed=1))
            W2 = tf.get_variable("W2", [filter_layer_2, filter_layer_2, channels_layer_1, channels_layer_2], initializer = tf.contrib.layers.xavier_initializer(seed=1))

        # CONV2D: stride of 1, padding 'SAME'
        Z1 = tf.nn.conv2d(self.X,W1, strides = [1,1,1,1], padding = 'SAME')
        # RELU
        A1 = tf.nn.relu(Z1)
        # MAXPOOL: window 8x8, sride 8, padding 'SAME'
        # P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
        # CONV2D: filters W2, stride 1, padding 'SAME'
        Z2 = tf.nn.conv2d(A1,W2, strides=[1,1,1,1], padding = 'SAME')
        # RELU
        A2 = tf.nn.relu(Z2)
        # MAXPOOL: window 4x4, stride 4, padding 'SAME'
        # P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides = [1,4,4,1], padding="SAME")
        # FLATTEN
        P2 = tf.contrib.layers.flatten(A2)
        # FULLY-CONNECTED without non-linear activation function (not not call softmax).
        # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"
        Z3 = tf.contrib.layers.fully_connected(P2, self.n_y, activation_fn=None)

        # Softmax outputs
        self.outputs_softmax = tf.nn.softmax(Z3, name='A3')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=self.Y)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def plot_cost(self):
        import matplotlib
        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
