import tensorflow as tf
# Hide some depreacation warnings and disable eager execution
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import sonnet as snt
from tqdm import tqdm
from graph_nets import graphs, utils_np, modules, utils_tf, blocks
import random
from maze_definition import maze_dict, batch_size_dict, n_epochs_dict
from utils import *
from GNN import *

# print('GPU ACTIVATED: ', tf.test.is_gpu_available())
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Use this to force Tensorflow on CPU

# DESCRIPTION: This file implements a Contrastively-trained Structured World Model (C-SWM) that can predict the state
# transition and reward given vectorized position vector of the [objects, target and agent] respectively and action.

root_dir = 'C:/Users/Yannick/PycharmProjects/RQ1/'
results_dir = root_dir + 'results/'

# set seed for re-producability:
random.seed(1)
np.random.seed(1)
# tf.set_random_seed(1)

#  REPLAY BUFFER CLASS
class ReplayBuffer:
    "see: https://spinningup.openai.com/en/latest/_modules/spinup/algos/ddpg/ddpg.html#ddpg"
    "Possible extensions (CER): https://arxiv.org/pdf/1712.01275.pdf"
    " (PER): https://arxiv.org/abs/1511.05952"
    CER = False


    # TODO: Extend with possibility of restoring sequences
    def __init__(self, obs_dim, act_dim, size, random_samples=True):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.size = size
        self.random_samples = random_samples
        self.obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.next_obs_buf = np.zeros([int(size), int(obs_dim)], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.rews_buf = np.zeros(int(size), dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=np.bool)
        self.sample_nr_buf = np.zeros(int(size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)


    def store(self, obs, act, rew, next_obs, done, sample_nr):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.sample_nr_buf[self.ptr] = sample_nr
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        #= np.copy(idxs)
        #np.random.shuffle(idxs2)
        #avoid end of episode states
        for i in range(len(idxs)):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        if self.random_samples == True:
            idxs2 = np.random.randint(0, self.size, size=idxs.shape[0])
        else:
            idxs2 = np.zeros_like(idxs)
            for k in range(len(idxs)):
                count = 0
                for j in range(100):
                    if self.done_buf[idxs[k] + 1 + j] == 1.0:
                        break
                    count = count + 1
                if count < 3:
                    idxs2[k] = np.random.randint(0, self.size, size=1)
                else:
                    idxs2[k] = np.random.choice(self.sample_nr_buf[idxs[k] + 2:idxs[k] + count], 1)[0].astype('int32')
        #for i in range(len(idxs2)):
        #    if idxs2[i] != 0:
        #        if self.done_buf[idxs2[i]-1] == 1.0:
        #            idxs2[i] = idxs2[i] + 3
        #        if self.done_buf[idxs2[i]-2] == 1.0:
        #            idxs2[i] = idxs2[i] + 2
        #        if self.done_buf[idxs2[i]-3] == 1.0:
        #            idxs2[i] = idxs2[i] + 1


        if self.CER: idxs[-1] = abs(self.ptr - 1) # this takes the last added sample, unless ptr = 0, then it takes sample 1, this then does violate CER

        return dict(obs1=self.obs_buf[idxs],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    obs3=self.obs_buf[idxs2])  # negative samples of state
                    # obs3=self.next_obs_buf[idxs2]) # negative samples of state

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    rews=self.rews_buf[:self.size],
                    done=self.done_buf[:self.size])
                    #sample_nr=self.sample_nr_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size)


class Contrastive_StructuredWorldModel:
    def __init__(self, obs_dim, act_dim, batch_size, n_objects, state_embedding_dim, reward_lr = 0.001,
                 transition_lr = 0.001, hidden_dim_forward_model = 32, hidden_dim_reward_model = 32, folder = 'results/',
                 random_samples=True, model_name = 'model'):

        self.act_dim = act_dim                                          # dimension of the action (one-hot encoded)
        self.obs_dim = obs_dim                                          # dimension of the observation (state) vector
        self.batch_size = batch_size                                    # batch size to use for the replay buffer
        self.state_dim_embedding = state_embedding_dim                  # dimension of the state embedding per object
        self.learning_rate_transition = transition_lr                   # learning rate for the transition model
        self.learning_rate_reward = reward_lr                           # learning rate for the reward model
        self.n_objects = n_objects                                      # one for the agent, one for the exit, object
        self.hidden_dim_transition_model = hidden_dim_forward_model     # hidden dimension forward model
        self.hidden_dim_reward_model = hidden_dim_reward_model          # hidden dimension reward model
        self.random_samples = random_samples                            # randomly sample in replay buffer
        self.l2_reg = 0.0                                               # weight normalization norm
        # normalization constants for the contrastive loss
        self.sigma = 0.5
        self.hinge = 1          # hinge

        self.model_folder = os.path.join(folder, 'trained_models', model_name)
        self.model_path = os.path.join(folder, 'trained_models', model_name, model_name + '.ckpt') # path where model is stored
        self.model_name = model_name                                    # name of the model

        self.graph = tf.Graph()
        with self.graph.as_default():
            # DEFINE PLACEHOLDERS
            self.obs_var = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="obs_var")                       # state
            self.next_obs_var = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="next_obs_var")             # next state
            self.neg_obs_var = tf.placeholder(tf.float32, shape=[None, self.obs_dim], name="neg_obs_var")               # negative sample
            self.action_var = tf.placeholder(tf.float32, shape=[None, self.act_dim], name="action_var")                 # action (one-hot)
            self.reward_var = tf.placeholder(tf.float32, shape=[None], name='reward_var')                               # reward
            self.is_training = tf.placeholder(tf.bool, shape=[], name="train_cond")                                     # training bool (Training -> True)

            # NN FORWARD (transition) MODEL AND REWARD MODEL
            self.forward_model_nn = snt.Module(self.forward_model_nn_network, name='Forward_nn_Network')
            self.reward_model_nn = snt.Module(self.reward_model_nn_network, name='Reward_nn_Network')
            self.delta_state_nn_pred = self.forward_model_nn(self.obs_var, self.action_var)
            self.reward_nn_pred = self.reward_model_nn(self.obs_var, self.action_var)

            # GRAPH RELATED PLACEHOLDERS
            # self.n_nodes_ph =tf.placeholder(tf.int32, shape=[None], name="n_nodes")
            # self.n_edges_ph = tf.placeholder(tf.int32, shape=[None], name="n_edges")
            # self.node_attributes_ph = tf.placeholder(tf.float32, shape=[None, self.n_objects, self.state_dim_embedding], name="node_attributes")

            # put the object encodings in a graph, nodes are the 2D position hopefully and the global can be the action
            self.obs_per_object = self.reshape_observation_vec_to_object_encoding(self.obs_var)
            self.obs_graph = self.object_encoder_graph(self.obs_per_object)
            self.next_obs_graph = self.object_encoder_graph(self.reshape_observation_vec_to_object_encoding(self.next_obs_var))

            # GNN FORWARD (TRANSITION) MODEL AND REWARD MODEL
            self.forward_model_gnn = snt.Module(self.forward_model_gnn_network, name='Forward_gnn_Network')
            self.reward_model_gnn = snt.Module(self.reward_model_gnn_network, name='Reward_gnn_Network')
            self.delta_state_gnn_pred = utils_tf.make_runnable_in_session(self.forward_model_gnn(self.obs_graph, self.action_var))
            self.reward_gnn_pred = self.reward_model_gnn(self.obs_graph, self.action_var)


            # MSE LOSS NN
            self.reward_loss_nn_mse = tf.reduce_mean(tf.squared_difference(self.reward_var, self.reward_nn_pred)) # MSE(r, r^)
            self.state_transition_loss_nn_mse = tf.reduce_mean(tf.squared_difference((self.delta_state_nn_pred + self.obs_var), self.next_obs_var)) # MSE(delta_s + s, s')

            # MSE LOSS GNN
            self.predicted_next_obs_graph = self.obs_graph # should we make a copy or something?
            self.predicted_next_obs_graph = self.predicted_next_obs_graph.replace(nodes = self.obs_graph.nodes + self.delta_state_gnn_pred.nodes) # sum node features to get the predicted next state
            self.state_transition_loss_gnn_mse = tf.reduce_mean(tf.squared_difference(self.predicted_next_obs_graph.nodes, self.next_obs_graph.nodes))
            self.reward_loss_gnn_mse = tf.reduce_mean(tf.squared_difference(self.reward_var, self.reward_gnn_pred))

            # # CONTRASTIVE LOSSES
            # # self.state_transition_nn_loss_hinge = tf.reduce_mean(
            # #     tf.maximum(0.0, self.hinge - 0.5 * tf.squared_difference(self.encoded_state + self.latent_action_prediction, self.encoded_state_2)))
            #
            # # SOME CONSTANTS:
            # self.norm = tf.constant((0.5 / self.sigma**2))
            # self.H = tf.reduce_mean(self.norm * tf.squared_difference((self.delta_state_nn_pred + self.encoded_state_vec), self.encoded_next_state_vec))  # batch_sum( d(s_t + delta s_t, s_t+1) )  eqn 5, first part
            # self.H_tilde = tf.reduce_mean(self.norm * tf.squared_difference(self.encoded_neg_state_vec, self.encoded_next_state_vec))  # batch_sum ( d(s_tilde, s_t+1) ) eqn 5, second part
            # self.state_transition_nn_loss_hinge = self.H + tf.maximum(0.0, (self.hinge - self.H_tilde))  # loss function as in eqn 6

            # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            # # self.train_nn_model = optimizer_reward.minimize(sum(self.reward_loss_nn_mse, self.state_transition_nn_loss))
            # self.train_fwd_model_op = optimizer.minimize(self.state_transition_nn_loss_hinge)

            # TRAINING - FORWARD MODEL NN
            optimizer_transition_nn = tf.train.AdamOptimizer(self.learning_rate_transition)
            self.train_op_fwd_model_nn = optimizer_transition_nn.minimize(self.state_transition_loss_nn_mse)
            # TRAINING - REWARD FUNCTION NN
            optimizer_reward_nn = tf.train.AdamOptimizer(self.learning_rate_reward)
            self.train_op_rwrd_model_nn = optimizer_reward_nn.minimize(self.reward_loss_nn_mse)

            # TRAINING - FORWARD MODEL GNN
            optimizer_transition_gnn = tf.train.AdamOptimizer(self.learning_rate_transition)
            self.train_op_fwd_model_gnn = optimizer_transition_gnn.minimize(self.state_transition_loss_gnn_mse)
            # TRAINING - REWARD FUNCTION NN
            optimizer_reward_gnn = tf.train.AdamOptimizer(self.learning_rate_reward)
            self.train_op_rwrd_model_gnn = optimizer_reward_gnn.minimize(self.reward_loss_gnn_mse)

            # SET THE REPLAY MEMORY FOR THE NETWORK --------------------------------------------------------------
            self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, 30000, random_samples = self.random_samples)

            # Init session --------------------------------------------------------------------------------------------
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

        # Initialize the session
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        # if not os.path.exists(os.path.join(self.model_folder, 'logs', model_name)):
        #     os.makedirs(os.path.join(self.model_folder, 'logs', model_name))
        # file_writer = tf.summary.FileWriter(os.path.join(self.model_folder, 'logs', model_name), self.sess.graph)

    ############################################# LEARN FUNCTIONS #############################################
    def learn_fwd_nn_model(self):
        batch = self.replay_buffer.sample_batch(self.batch_size) # sample batch

        feed_dict = {self.obs_var: batch['obs1'],
                     self.next_obs_var: batch['obs2'],
                     self.action_var: batch['acts'],
                     self.reward_var: batch['rews'],
                     self.is_training: True
                     }

        _, transition_loss = self.sess.run([self.train_op_fwd_model_nn, self.state_transition_loss_nn_mse], feed_dict)
        return transition_loss # transtition loss

    def learn_rwrd_nn_model(self):
        batch = self.replay_buffer.sample_batch(self.batch_size) # sample batch

        feed_dict = {self.obs_var: batch['obs1'],
                     self.action_var: batch['acts'],
                     self.reward_var: batch['rews'],
                     self.is_training: True
                     }

        _, reward_loss = self.sess.run([self.train_op_rwrd_model_nn, self.reward_loss_nn_mse], feed_dict)
        return reward_loss # reward loss

    def learn_fwd_gnn_model(self):
        batch = self.replay_buffer.sample_batch(self.batch_size) # sample batch

        feed_dict = {self.obs_var: batch['obs1'],
                     self.next_obs_var: batch['obs2'],
                     self.action_var: batch['acts'],
                     self.is_training: True
                     }

        _, transition_loss = self.sess.run([self.train_op_fwd_model_gnn, self.state_transition_loss_gnn_mse], feed_dict)
        return transition_loss # transtition loss

    def learn_rwrd_gnn_model(self):
        batch = self.replay_buffer.sample_batch(self.batch_size) # sample batch

        feed_dict = {self.obs_var: batch['obs1'],
                     self.action_var: batch['acts'],
                     self.reward_var: batch['rews'],
                     self.is_training: True
                     }

        _, reward_loss = self.sess.run([self.train_op_rwrd_model_gnn, self.reward_loss_gnn_mse], feed_dict)
        return reward_loss # reward loss

    ############################################# PREDICT FUNCTIONS #############################################
    def predict_object_encodings(self, state_vec):
        out = self.sess.run(self.obs_per_object, feed_dict={self.obs_var: state_vec})
        return out

    def predict_state_graph(self, state_vec):
        # node_attributes = self.predict_object_encodings(state_vec)
        out = self.sess.run(self.obs_graph, feed_dict={self.obs_var: state_vec})
        return out

    def predict_next_state_nn(self, z, action):
        # z = self.sess.run(self.encoded_state_vec, feed_dict={self.obs_var: image, self.is_training: False})
        delta_z = self.sess.run(self.delta_state_nn_pred, feed_dict={self.obs_var: z,
                                                                     self.action_var: action})
        next_z = z + delta_z
        return next_z

    def predict_reward_nn(self, z, action):
        reward = self.sess.run(self.reward_nn_pred, feed_dict={self.obs_var: z, self.action_var: action})
        return reward

    # def predict_next_obs_graph(self, z, action):
    #     return self.sess.run(self.predicted_next_obs_graph, feed_dict={self.obs_var: z, self.action_var: action})

    def predict_next_state_gnn2(self, z, action):
        delta_z = self.sess.run(self.delta_state_gnn_pred, feed_dict={self.obs_var: z,
                                                                     self.action_var: action})
        next_z = z + np.reshape(delta_z.nodes, newshape=(self.n_objects * self.state_dim_embedding))
        return next_z

    def predicht_reward_gnn(self, z, action):
        reward = self.sess.run(self.reward_gnn_pred, feed_dict={self.obs_var: z, self.action_var: action})
        return reward

    # def predict_next_state_gnn(self, z, action):
    #     # a = utils_tf.make_runnable_in_session(z_graph)
    #     # test = utils_tf.get_feed_dict(self.obs_graph, a)
    #     delta_z = self.sess.run(self.delta_state_gnn_pred, feed_dict={self.obs_var: z,
    #                                                                  self.action_var: action})
    #     return delta_z

    ############################################# Manual Object Encoding from Observation Vector Representation #############################################
    def reshape_observation_vec_to_object_encoding(self, enc_obs_vec):
        return tf.reshape(enc_obs_vec, shape = (-1, self.n_objects, self.state_dim_embedding))

    ############################################# Graph State Representation #############################################
    def object_encoder_graph(self, encodings_per_object):
        """Function to make a graph object from the object oriented encoding
        Inputs:
            encodings_per_object: tensor containing the embeddings per object slot (size : [batch, n_objects, embedding dim])
        Outputs:
            graph_representation: fully conected graph with as node attributes the object encodings (size : [batch, graph])
        """
        # Specify the number of nodes and edges implied by to the passed n_objects and the encodigs for each object
        n_nodes = tf.tile(tf.constant([self.n_objects]), tf.shape(encodings_per_object)[0:1], name='n_nodes')
        n_edges = tf.tile(tf.constant([0]), tf.shape(encodings_per_object)[0:1], name='n_edges')
        # put the node_attributes in the correct shape to make graph object:
        node_attributes = tf.reshape(encodings_per_object, [tf.shape(encodings_per_object)[0] * tf.shape(encodings_per_object)[1], self.state_dim_embedding])
        # make graph object with specified node attributes:
        graph = graphs.GraphsTuple(nodes=node_attributes, edges=None, globals=None,
                                                  receivers=None, senders=None, n_node=n_nodes, n_edge=n_edges)
        # Connect all node to the other nodes (i.e. make fully connected):
        fully_connected_graph = utils_tf.fully_connect_graph_dynamic(graph, exclude_self_edges=False)
        # Make it runnable in TF because None's are used:
        runnable_fc_graph = utils_tf.make_runnable_in_session(fully_connected_graph)
        return runnable_fc_graph

    ############################################# Transition Model - NN #############################################
    def forward_model_nn_network(self, encoded_state_vec, action, l2_reg = 0.0):
        """NN-based transition function."""
        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l2_reg)}
        initializers = {"w": tf.keras.initializers.he_normal()}
        # action [Left, No-op, No-op]
        # action [1 0 0 0], [0 0 0 0], [0 0 0 0]
        # concatenate action -> how to force this on the objects?

        # flat_state = tf.layers.flatten(encoded_state_vec)
        concate_state_action = tf.concat([encoded_state_vec, action], axis = 1, name = 'concate_input')

        # hidden layer 1 - RELU
        fc1 = tf.layers.dense(inputs = concate_state_action, units = self.hidden_dim_transition_model, activation = tf.nn.relu,
                              kernel_initializer=initializers["w"]) # kernel_regularizer=regularizers["w"])
        # hidden layer 2 - No activation
        fc2 = tf.layers.dense(inputs=fc1, units=self.hidden_dim_transition_model, activation=tf.nn.relu,
                             kernel_initializer=initializers["w"]) # kernel_regularizer=regularizers["w"]
        # LayerNorm
        ln = tf.contrib.layers.layer_norm(fc2)
        # Object encoding
        fc2 = tf.nn.relu(ln)
        # output layer
        state_transition = tf.layers.dense(inputs=fc2, units=self.n_objects * self.state_dim_embedding, activation=None,
                              kernel_regularizer=regularizers["w"], kernel_initializer=initializers["w"])
        return state_transition

    ############################################# Reward Model - NN #############################################
    def reward_model_nn_network(self, encoded_state_vec, action, l2_reg = 0.0):
        """NN-based reward function.
        Inputs:
            encoded_state_vec: tensor containing the flattened object embeddings
            action: one hot encoded action
        Outputs:
            reward: scalar value as reward
        """
        regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=l2_reg)}
        initializers = {"w": tf.keras.initializers.he_normal()}

        # flat_state = tf.layers.flatten(encoded_state_vec)
        concate_state_action = tf.concat([encoded_state_vec, action], axis = 1)

        # hidden layer 1 - RELU
        fc1 = tf.layers.dense(inputs = concate_state_action, units = self.hidden_dim_transition_model, activation = tf.nn.relu,
                              kernel_initializer=initializers["w"]) # kernel_regularizer=regularizers["w"])
        # hidden layer 2 - No activation
        fc2 = tf.layers.dense(inputs=fc1, units=self.hidden_dim_transition_model, activation=tf.nn.relu,
                             kernel_initializer=initializers["w"]) # kernel_regularizer=regularizers["w"]
        # LayerNorm
        ln = tf.contrib.layers.layer_norm(fc2)
        # Object encoding
        fc2 = tf.nn.relu(ln)
        # output layer
        reward_prediction = tf.layers.dense(inputs=fc2, units=1, activation=None,
                              kernel_regularizer=regularizers["w"]) #kernel_initializer=initializers["w"])
        return reward_prediction

    ############################################# Transition Model - GNN #############################################
    def forward_model_gnn_network(self, encoded_state_graph, action, l2_reg = 0.0):
        state_action_graph = encoded_state_graph.replace(globals = action) # put the action as global

        fwd_model_gnn = GraphNeuralNetwork_transition(edge_model_fn=lambda: make_mlp_model(hidden_dim=self.hidden_dim_transition_model, out_dim=self.hidden_dim_transition_model),
                                                      node_model_fn=lambda: make_mlp_model(hidden_dim=self.hidden_dim_transition_model, out_dim=self.state_dim_embedding))

        state_transition = fwd_model_gnn(state_action_graph)
        return state_transition

    ############################################# Reward Model - GNN #############################################
    def reward_model_gnn_network(self, encoded_state_graph, action, l2_reg = 0.0):
        state_action_graph = encoded_state_graph.replace(globals = action) # put the action as global

        rwrd_model_gnn = GraphNeuralNetwork_reward(node_model_fn=lambda: make_mlp_model(hidden_dim=self.hidden_dim_reward_model, out_dim=self.hidden_dim_reward_model),
                                                   global_model_fn=lambda: make_mlp_model(hidden_dim=self.hidden_dim_reward_model, out_dim=1))

        reward = rwrd_model_gnn(state_action_graph)
        return reward

    #############################################  #############################################
    def object_decoder_network(self):
        pass


    ############################################# Contrastive Loss #############################################
    # def contrastive_loss(self):
    #     objects = self.object_extractor_nn(self.obs_var, self.is_training, self.l2_reg)
    #     next_objects = self.object_extractor_nn(self.next_obs_var, self.is_training, self.l2_reg)
    #
    #     state = self.object_encoder_nn(objects, self.l2_reg)
    #     next_state = self.object_encoder_nn(next_objects, self.l2_reg)
    #
    #     # negative samples
    #     neg_state = tf.random.shuffle(state, seed=None, name=None) # DOES THIS WORK LIKE THIS???
    #
    #     delta_state = self.transition_model_nn_network()
    #
    #     # SOME CONSTANTS:
    #     self.gamma = tf.constant(self.hinge)
    #     self.zero = tf.constant(0.0)
    #     self.norm = tf.constant((0.5 / self.sigma ** 2))
    #     # batch_sum( d(s_t + delta s_t, s_t+1) )  eqn 5, first part
    #     self.H = tf.reduce_mean(tf.math.multiply(self.norm,tf.squared_difference((self.predicted_state_diff + self.state),
    #                                                                              self.next_state)))
    #     # batch_sum ( d(s_tilde, s_t+1) ) eqn 5, second part
    #     self.H_tilde = tf.reduce_mean(tf.math.multiply(self.norm, tf.squared_difference(self.neg_state_ph, self.next_state_ph)))
    #
    #     self.loss_transition = tf.math.add(self.H, tf.maximum(self.zero, (
    #                 self.gamma_tf - self.H_tilde)))  # loss function as in eqn 6
    #
    #     return loss

    def remember(self, state, action, reward, episode_start, next_state, done):
        """ simply adds a experience tuple to the replay memory """
        self.replay_buffer.store(state, action, reward, episode_start, next_state, done)

    # def set_model_path(self, folder, filename):
    #     self.model_path = os.path.join(folder, filename + ".ckpt")
    #     print(self.model_path)

    def save_model(self):
        """ saves all variables in the session to memory """
        print("Storing Model...")
        # make sure the folder exists
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        # save data
        return self.saver.save(self.sess, self.model_path)

    def load_model(self):
        """ loads all variables in the session from memory """
        print("Restoring Model...")
        self.saver.restore(self.sess, self.model_path)



# MAIN FUN
if __name__ == '__main__':
    from Logger import Logger
    # import Plotter as plotter
    # import plotter_2

    for maze in maze_dict:
        # maze = 'maze_5x5_2_rooms'  # the maze to use, TODO: make this iterative through all mazes
        print('Maze to learn model for: {}'.format(maze))

        folder_data = os.path.join(results_dir, maze)
        data_true = load_pickle(os.path.join(folder_data, 'true_log' + '.pkl')) # the log with the true position of the agent
        data_observation = load_pickle(os.path.join(folder_data, 'observation_log' + '.pkl')) # the log with the observation vector of the maze (environment)
        print('Loaded {} data points'.format(len(data_observation)))

        n_epochs = n_epochs_dict[maze]          # number of epochs for trianing nn model TODO: make this scale with the used maze
        # n_epochs_gnn = 10                     # number of epochs for trianing gnn model TODO: make this scale with the used maze
        batch_size = batch_size_dict[maze]      # TODO: make this scale with the used maze
        lr_transition_model_nn = 0.001          # learning rates nn model
        lr_reward_model_nn = 0.001
        lr_transition_model_gnn = 5e-3        # learning rate gnn model
        lr_reward_model_gnn = 5e-5

        observation_size = int(len(data_observation[0][0]))
        action_size = 4                                 # 4 possible actions : north east south west
        hinge = 1.0
        random_samples = True
        n_object_slots = int(observation_size / 2)      # make this equal to the len of the observation vector /2
        latent_state_size = 2                           # corresponding to the position of the objects


    # [forward_loss, reward_loss, contrastive_loss]
    # loss_coef = [1.0, 1.0, 1.0]

        cswm_nn = Contrastive_StructuredWorldModel(obs_dim=observation_size, act_dim=action_size, batch_size=batch_size,
                                                n_objects=n_object_slots, state_embedding_dim=latent_state_size,
                                                reward_lr = lr_reward_model_nn, transition_lr = lr_transition_model_nn,
                                                hidden_dim_forward_model = 32, hidden_dim_reward_model=32,
                                                folder=folder_data, model_name='nn_model', random_samples= True)

        cswm_gnn = Contrastive_StructuredWorldModel(obs_dim=observation_size, act_dim=action_size, batch_size=batch_size,
                                                n_objects=n_object_slots, state_embedding_dim=latent_state_size,
                                                reward_lr = lr_reward_model_gnn, transition_lr = lr_transition_model_gnn,
                                                hidden_dim_forward_model = 32, hidden_dim_reward_model=32,
                                                folder=folder_data, model_name='gnn_model', random_samples= True)
        # add all data to srl memory
        print('Load data into replay buffer...')
        for d in tqdm(data_observation):
                # cswm.remember(rescale(d[0]).astype('float32'), d[1], np.round(d[2], decimals=3), rescale(d[3]).astype('float32'), d[4], d[5])
                cswm_nn.remember(d[0].astype('float32'), one_hot_encoded_action(action=d[1], n_actions=action_size),
                              np.round(d[2], decimals=3), d[3].astype('float32'), d[4], d[5])

                cswm_gnn.remember(d[0].astype('float32'), one_hot_encoded_action(action=d[1], n_actions=action_size),
                                 np.round(d[2], decimals=3), d[3].astype('float32'), d[4], d[5])

        ########## SOME TESTS ############
        batch = cswm_nn.replay_buffer.sample_batch(3)
        # print('test data flow using batch samples')
        #
        # # test reshaping object encodings:
        # object_encoding_test = cswm_nn.predict_object_encodings(batch['obs1'])
        # # test object encoding as graph:
        # state_graph = cswm_gnn.predict_state_graph(batch['obs1'])
        # print('Object encoding as graph')
        # graphs_nx = utils_np.graphs_tuple_to_networkxs(state_graph)
        # _, axes = plt.subplots(1, 1, figsize=(5, 5))
        # plot_graph_networkx(graphs_nx[0], axes)

        # test_vec = cswm_gnn.predict_next_state_nn(batch['obs1'], np.array([[0,0,0,1],[0,1,0,0],[0,0,0,1]]))
        # test2 = cswm_gnn.predict_next_state_gnn2(np.array([batch['obs1'][0]]), np.array([[0,0,0,1]]))
        # test_graph = cswm_gnn.predict_next_state_gnn(batch['obs1'], np.array([[0,0,0,1],[0,1,0,0],[0,0,0,1]]))


        ########### NN TRAINING LOOP: ############
        print('Start Training: NN model')
        # initialize loss:
        loss_history_transition_nn = []
        loss_history_reward_nn = []

        for epoch in range(n_epochs): # EPOCH LOOP
            loss_episode_transition_nn = 0 # init episode loss
            loss_episode_reward_nn = 0

            for batch_number in range(int(len(data_observation)/batch_size)):
                loss_fwd_nn = cswm_nn.learn_fwd_nn_model()
                loss_rwrd_nn = cswm_nn.learn_rwrd_nn_model()
                loss_episode_transition_nn += loss_fwd_nn
                loss_episode_reward_nn += loss_rwrd_nn

            if epoch % 2 == 0:
                # print("Epoch: {} \t Transition Loss NN: {} \t Reward Loss: {}".format(epoch,loss_episode_transition_nn, loss_episode_reward_nn))
                print("Maze: {} \t Epoch: {}/{} \t Transition Loss NN: {} \t Reward Loss NN: {}".format(maze, epoch, n_epochs, np.round(loss_episode_transition_nn,4) , np.round(loss_episode_reward_nn,4)))

            loss_history_transition_nn.append(loss_episode_transition_nn)
            loss_history_reward_nn.append(loss_episode_reward_nn)
        cswm_nn.save_model() # save model


        ########### GNN TRAINING LOOP: ############
        print('Start Training: GNN model')
        # initialize loss:
        loss_history_transition_gnn = []
        loss_history_reward_gnn = []

        for epoch in range(n_epochs):  # EPOCH LOOP
            loss_episode_transition_gnn = 0 # init episode loss
            loss_episode_reward_gnn = 0

            for batch_number in range(int(len(data_observation) / batch_size)):
                loss_fwd_gnn = cswm_gnn.learn_fwd_gnn_model()
                loss_rwrd_gnn = cswm_gnn.learn_rwrd_gnn_model()

                # print(loss_fwd_nn)
                loss_episode_transition_gnn += loss_fwd_gnn
                loss_episode_reward_gnn += loss_rwrd_gnn

            if epoch % 2 == 0:
                print("Maze: {} \t Epoch: {}/{} \t Transition Loss GNN: {} \t Reward Loss GNN: {}".format(maze, epoch, n_epochs, np.round(loss_episode_transition_gnn,4) , np.round(loss_episode_reward_gnn,4)))

            loss_history_transition_gnn.append(loss_episode_transition_gnn)
            loss_history_reward_gnn.append(loss_episode_reward_gnn)
        cswm_gnn.save_model()  # save model




        ############# PLOTS #############
        plt.figure()
        plt.plot(np.arange(len(loss_history_transition_nn)), loss_history_transition_nn)
        plt.plot(np.arange(len(loss_history_reward_nn)), loss_history_reward_nn)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['forward model', 'reward model'])
        plt.grid()
        x1, x2, y1, y2 = plt.axis() # obtain current axis limits
        plt.axis((x1, x2, 0, min(max(loss_history_reward_nn), max(loss_history_transition_gnn)))) # set y axis to the max of both error histories
        filename = folder_data + '/trained_models/' + 'nn_model_loss_curves' + '.pdf'
        plt.savefig(filename)

        plt.figure()
        plt.plot(np.arange(len(loss_history_transition_gnn)), loss_history_transition_gnn)
        plt.plot(np.arange(len(loss_history_reward_gnn)), loss_history_reward_gnn)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(['forward model', 'reward model'])
        plt.grid()
        x1, x2, y1, y2 = plt.axis() # obtain current axis limits
        plt.axis((x1, x2, 0, min(max(loss_history_reward_nn), max(loss_history_transition_gnn)))) # set y axis to the max of both error histories
        filename = folder_data + '/trained_models/' + 'gnn_model_loss_curves' + '.pdf'
        plt.savefig(filename)

    print('end')

