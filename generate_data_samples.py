# generate data samples that are used for training the transition and reward models

# IMPORTS
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# CUSTOM IMPORTS
from Logger import Logger
from maze_definition import maze_dict
from utils import *
from environment.maze import Maze, Status


# HYPERPARAMETERS
start_pos = (0, 0)                  # default start position agent
n_actions = 4                       # number of actions
folder_name = 'results'
# random_start = True
# random_target = True
save_environment_plot_every_episode = True
show_environment_every_episode = False          # plot the environment of every episode while creating the data samples
# n_episodes = 10
max_iterations_episode = 1000                     # max number of steps the agent can make before forced termination

# set seed for re-producability:
random.seed(1)
np.random.seed(1)
tf.set_random_seed(1)


if __name__ == '__main__':
    for maze_name, maze in maze_dict.items(): # loop through different mazes
        logging_folder = folder_name + '/' + maze_name + '/'
        true_logger = Logger(folder=logging_folder, filename='true_log.pkl')  # create logger for true state
        observation_logger = Logger(folder=logging_folder, filename='observation_log.pkl')  # create logger

        y_max, x_max = maze.shape
        target_pos = (x_max - 1, y_max - 1)  # default target position agent
        n_rows = y_max
        n_cols = x_max

        n_episodes = n_rows * n_cols    # set the n_episodes equal to the number of cells in the maze
        complete_episodes = 0           # completed number of episodes
        sample_number = 0               # sample number count

        print('Start generating samples for maze: ' + maze_name)
        print('Total number of episodes: {}'.format(n_episodes))
        for episode in tqdm(range(n_episodes)):
            # create environment object with specified start and target position
            start_pos, target_pos = choose_random_start_and_target(maze) # get random starting and target position
            environment = Maze(maze, start_cell=start_pos, exit_cell=target_pos)

            if save_environment_plot_every_episode: # save environment layout every episode
                file_format = '.png'
                folder_images = 'maze_layouts/'
                file_name = maze_name + '_' + 'episode_' + str(episode) + file_format
                environment.plot_environment(show_plot=show_environment_every_episode, folder=logging_folder + '/' + folder_images, filename=file_name)

            Terminate = False
            cnt = 0
            state = np.array(start_pos) # set starting state

            while True: # walk in the maze until termination
                action = np.random.choice(environment.actions)                  # choose random action
                observation_vec = environment.create_observation_vec()          # make state vector representationwith position of all objects
                next_state, reward, status = environment.step(action)           # perform step in maze
                next_observation_vec = environment.create_observation_vec()     # make next state vector representationwith position of all objects

                if status == Status.WIN:  # terminal state reached
                    done = True
                else:
                    done = False

                # save true state, action, reward, next state, done and episode number to logger file
                true_logger.obslog((state, action, reward, next_state, done, sample_number))

                # save observatino, action, reward, next observation, done and episode number to logger file
                observation_logger.obslog((observation_vec, action, reward, next_observation_vec, done, sample_number))

                # set state to next state
                state = next_state

                if status == Status.WIN:            # terminal state reached
                    complete_episodes += 1
                    print('\nEpisode terminated because goal state was reached')
                    break
                if cnt == max_iterations_episode:   # terminate because took too long
                    print('\nEpisode terminated because it took too long...')
                    break
                cnt += 1
                sample_number += 1

        print("Finished generating samples for maze: " + maze_name)
        print("Reached the goal state {}/{}".format(complete_episodes, n_episodes))

        true_logger.save_obslog()
        observation_logger.save_obslog()