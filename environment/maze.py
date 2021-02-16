import logging
import random
from enum import Enum, IntEnum
import matplotlib.pyplot as plt
import numpy as np
import os


# Code used from: https://github.com/erikdelange/Reinforcement-Learning-Maze

class Cell(IntEnum):
    EMPTY = 0  # indicates empty cell where the agent can move to
    OCCUPIED = 1  # indicates cell which contains a wall and cannot be entered
    CURRENT = 2  # indicates current cell of the agent


class Action(IntEnum):
    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3


class Render(Enum):
    NOTHING = 0
    TRAINING = 1
    MOVES = 2


class Status(Enum):
    WIN = 0
    LOSE = 1
    PLAYING = 2


class Maze:
    """ A maze with walls. An agent is placed at the start cell and must find the exit cell by moving through the maze.

        The layout of the maze and the rules how to move through it are called the environment. An agent is placed
        at start_cell. The agent chooses actions (move left/right/up/down) in order to reach the exit_cell. Every
        action results in a reward or penalty which are accumulated during the game. Every move gives a small
        penalty (-0.05), returning to a cell the agent visited earlier a bigger penalty (-0.25) and running into
        a wall a large penalty (-0.75). The reward (+10.0) is collected when the agent reaches the exit. The
        game always reaches a terminal state; the agent either wins or looses. Obviously reaching the exit means
        winning, but if the penalties the agent is collecting during play exceed a certain threshold the agent is
        assumed to wander around clueless and looses.

        A note on cell coordinates:
        The cells in the maze are stored as (col, row) or (x, y) tuples. (0, 0) is the upper left corner of the maze.
        This way of storing coordinates is in line with what matplotlib's plot() function expects as inputs. The maze
        itself is stored as a 2D numpy array so cells are accessed via [row, col]. To convert a (col, row) tuple
        to (row, col) use: (col, row)[::-1]
    """
    actions = [Action.MOVE_LEFT, Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_DOWN]  # all possible actions

    reward_exit = 0 # reward for reaching the exit cell
    penalty_move = -1  # penalty for a move which did not result in finding the exit cell
    penalty_visited = -1  # penalty for returning to a cell which was visited earlier
    penalty_impossible_move = -1  # penalty for trying to enter an occupied cell or moving out of the maze

    def __init__(self, maze, start_cell=(0, 0), exit_cell=None):
        """ Create a new maze game.

            :param numpy.array maze: 2D array containing empty cells (= 0) and cells occupied with walls (= 1)
            :param tuple start_cell: starting cell for the agent in the maze (optional, else upper left)
            :param tuple exit_cell: exit cell which the agent has to reach (optional, else lower right)
        """
        self.maze = maze

        self.__minimum_reward = - 100 * self.maze.size  # stop game if accumulated reward is below this threshold
        # print(self.__minimum_reward)
        nrows, ncols = self.maze.shape
        self.nrows, self.ncols = self.maze.shape
        self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
        self.__exit_cell = (ncols - 1, nrows - 1) if exit_cell is None else exit_cell
        self.empty.remove(self.__exit_cell)  # removes the exit tile from the list of empty tiles (free tiles)

        # Check for impossible maze layout
        if self.__exit_cell not in self.cells:
            raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
        if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))

        # Variables for rendering
        self.__render = Render.NOTHING  # what to render
        self.__ax1 = None  # axes for rendering the moves
        self.__ax2 = None  # axes for rendering the best action per cell

        self.reset(start_cell)

    def test(self, ncols):
        print(ncols)

    def get_empty_cells(self):
        return self.empty

    def random_maze(self, maze, start=None, end=None):
        cells = [(col, row) for col in range(ncols) for row in range(nrows)]
        empty_cells = [(col, row) for col in range(ncols) for row in range(nrows) if maze[row, col] == Cell.EMPTY]
        # print(cells)
        # print(empty_cells)
        if start is None:
            start = random.choice(empty_cells)
            empty_cells.remove(start)
        else:
            pass  # use start

        if end is None:
            end = random.choice(empty_cells)
            empty_cells.remove(end)
        else:
            pass  # use end
        # print(maze,start,end)
        self.__init__(maze, start_cell=start, exit_cell=end)

    def set_exit_pos(self, exit_cell=None):
        if exit_cell is not self.__exit_cell:
            self.empty.append(self.__exit_cell)  # add previous cell to list of free cells
            self.__exit_cell = exit_cell  # set the new exit cell
            self.empty.remove(self.__exit_cell)  # removes the exit tile from the list of empty tiles (free tiles)

            # Check for impossible maze layout
            if self.__exit_cell not in self.cells:
                raise Exception("Error: exit cell at {} is not inside maze".format(self.__exit_cell))
            if self.maze[self.__exit_cell[::-1]] == Cell.OCCUPIED:
                raise Exception("Error: exit cell at {} is not free".format(self.__exit_cell))
        else:
            self.__exit_cell = (self.ncols - 1, self.nrows - 1)

        return self.__exit_cell

    # def shuffle_maze(self):
    #     maze_shuffled = self.maze.copy()
    #     maze_shuffled = maze_shuffled.ravel() # make 1d
    #     np.random.shuffle(maze_shuffled) # shuffle
    #     maze_shuffled = maze_shuffled.reshape(maze.shape) # make original shape
    #     print(maze_shuffled)
    #     self.maze = maze_shuffled
    #     self.cells = [(col, row) for col in range(ncols) for row in range(nrows)]
    #     self.empty = [(col, row) for col in range(ncols) for row in range(nrows) if self.maze[row, col] == Cell.EMPTY]
    #     print(self.empty)

    def reset(self, start_cell=(0, 0)):
        """ Reset the maze to its initial state and place the agent at start_cell.

            :param tuple start_cell: here the agent starts its journey through the maze (optional, else upper left)
            :return: new state after reset
        """
        if start_cell not in self.cells:
            raise Exception("Error: start cell at {} is not inside maze".format(start_cell))
        if self.maze[start_cell[::-1]] == Cell.OCCUPIED:
            raise Exception("Error: start cell at {} is not free".format(start_cell))
        if start_cell == self.__exit_cell:
            raise Exception("Error: start- and exit cell cannot be the same {}".format(start_cell))

        self.__previous_cell = self.__current_cell = start_cell
        self.__total_reward = 0.0  # accumulated reward
        self.__visited = set()  # a set() only stores unique values

        if self.__render in (Render.TRAINING, Render.MOVES):
            # render the maze
            nrows, ncols = self.maze.shape
            self.__ax1.clear()
            # self.__ax1.set_xticks(np.arange(0.5, nrows, step=1))
            # self.__ax1.set_xticklabels([])
            self.__ax1.set_xticks(np.arange(0, nrows, step=1))
            self.__ax1.set_xticks(np.arange(0.5, nrows, step=1), minor=True)
            self.__ax1.set_xticklabels(np.arange(0, ncols, 1))

            self.__ax1.set_yticks(np.arange(0, ncols, step=1))
            self.__ax1.set_yticks(np.arange(0.5, ncols, step=1), minor=True)
            self.__ax1.set_yticklabels(np.arange(0, nrows, 1))

            self.__ax1.grid(True, 'minor')
            self.__ax1.plot(*self.__current_cell, "rs", markersize=48)  # start is a big red square
            self.__ax1.text(*self.__current_cell, "Start", ha="center", va="center", color="white")

            self.__ax1.plot(*self.__exit_cell, "gs", markersize=48)  # exit is a big green square
            self.__ax1.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")
            self.__ax1.imshow(self.maze, cmap="binary")
            self.__ax1.get_figure().canvas.draw()
            self.__ax1.get_figure().canvas.flush_events()

        return self.__observe()

    def __draw(self):
        """ Draw a line from the agents previous cell to its current cell. """
        self.__ax1.plot(*zip(*[self.__previous_cell, self.__current_cell]), "bo-")  # previous cells are blue dots
        self.__ax1.plot(*self.__current_cell, "ro")  # current cell is a red dot
        self.__ax1.get_figure().canvas.draw()
        self.__ax1.get_figure().canvas.flush_events()

    def render(self, content=Render.NOTHING):
        """ Record what will be rendered during play and/or training.

            :param Render content: NOTHING, TRAINING, MOVES
        """
        self.__render = content

        if self.__render == Render.NOTHING:
            if self.__ax1:
                self.__ax1.get_figure().close()
                self.__ax1 = None
            if self.__ax2:
                self.__ax2.get_figure().close()
                self.__ax2 = None
        if self.__render == Render.TRAINING:
            if self.__ax2 is None:
                fig, self.__ax2 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Best move")
                self.__ax2.set_title("Best move")
                self.__ax2.set_axis_off()
                self.render_q(None)
        if self.__render in (Render.MOVES, Render.TRAINING):
            if self.__ax1 is None:
                fig, self.__ax1 = plt.subplots(1, 1, tight_layout=True)
                fig.canvas.set_window_title("Maze")
                # self.__ax1.set_title("Maze")

        plt.show(block=False)

    def step(self, action):
        """ Move the agent according to 'action' and return the new state, reward and game status.

            :param Action action: the agent will move in this direction
            :return: state, reward, status
        """
        reward = self.__execute(action)
        self.__total_reward += reward
        status = self.__status()
        state = self.__observe()
        logging.debug("action: {:10s} | reward: {: .2f} | status: {}".format(Action(action).name, reward, status))
        return state, reward, status

    def __execute(self, action):
        """ Execute action and collect the reward or penalty.

            :param Action action: direction in which the agent will move
            :return float: reward or penalty which results from the action
        """
        possible_actions = self.__possible_actions(self.__current_cell)

        if not possible_actions:
            # reward = -1000
            reward = self.__minimum_reward - 1  # cannot move anywhere, force end of game
        elif action in possible_actions:
            col, row = self.__current_cell
            if action == Action.MOVE_LEFT:
                col -= 1
            elif action == Action.MOVE_UP:
                row -= 1
            if action == Action.MOVE_RIGHT:
                col += 1
            elif action == Action.MOVE_DOWN:
                row += 1

            self.__previous_cell = self.__current_cell
            self.__current_cell = (col, row)

            if self.__render != Render.NOTHING:
                self.__draw()

            if self.__current_cell == self.__exit_cell:
                reward = Maze.reward_exit  # maximum reward when reaching the exit cell
            elif self.__current_cell in self.__visited:
                reward = Maze.penalty_visited  # penalty when returning to a cell which was visited earlier
            else:
                reward = Maze.penalty_move  # penalty for a move which did not result in finding the exit cell

            self.__visited.add(self.__current_cell)
        else:
            reward = Maze.penalty_impossible_move  # penalty for trying to enter an occupied cell or move out of the maze

        return reward

    def possible_next_states(self, action):
        """ Return the possible next state, given an action
        """

    def __possible_actions(self, cell=None):
        """ Create a list with all possible actions from 'cell', avoiding the maze's edges and walls.

            :param tuple cell: location of the agent (optional, else use current cell)
            :return list: all possible actions
        """
        if cell is None:
            col, row = self.__current_cell
        else:
            col, row = cell

        possible_actions = Maze.actions.copy()  # initially allow all

        # now restrict the initial list by removing impossible actions
        nrows, ncols = self.maze.shape
        if row == 0 or (row > 0 and self.maze[row - 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_UP)
        if row == nrows - 1 or (row < nrows - 1 and self.maze[row + 1, col] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_DOWN)

        if col == 0 or (col > 0 and self.maze[row, col - 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_LEFT)
        if col == ncols - 1 or (col < ncols - 1 and self.maze[row, col + 1] == Cell.OCCUPIED):
            possible_actions.remove(Action.MOVE_RIGHT)

        return possible_actions

    def __status(self):
        """ Return the game status.

            :return Status: current game status (WIN, LOSE, PLAYING)
        """
        if self.__current_cell == self.__exit_cell:
            return Status.WIN

        if self.__total_reward < self.__minimum_reward:  # force end of game after to much loss
            # print('LOST!')
            return Status.LOSE

        return Status.PLAYING

    def __observe(self):
        """ Return the state of the maze - in this game the agents current location.

            :return numpy.array [1][2]: agents current location
        """
        return np.array([*self.__current_cell])

    def play(self, model, start_cell=(0, 0)):
        """ Play a single game, choosing the next move based a prediction from 'model'.

            :param class AbstractModel model: the prediction model to use
            :param tuple start_cell: agents initial cell (optional, else upper left)
            :return Status: WIN, LOSE
        """
        self.reset(start_cell)

        state = self.__observe()

        while True:
            action = model.predict(state=state)
            state, reward, status = self.step(action)
            if status in (Status.WIN, Status.LOSE):
                return status

    def check_win_all(self, model):
        """ Check if the model wins from all possible starting cells. """
        previous = self.__render
        self.__render = Render.NOTHING  # avoid rendering anything during execution of the check games

        win = 0
        lose = 0

        for cell in self.empty:
            if self.play(model, cell) == Status.WIN:
                win += 1
            else:
                lose += 1

        self.__render = previous  # restore previous rendering setting

        logging.info("won: {} | lost: {} | win rate: {:.5f}".format(win, lose, win / (win + lose)))

        result = True if lose == 0 else False

        return result, win / (win + lose)

    def render_q(self, model):
        """ Render the recommended action(s) for each cell as provided by 'model'.

        :param class AbstractModel model: the prediction model to use
        """

        def clip(n):
            return max(min(1, n), 0)

        if self.__render == Render.TRAINING:
            nrows, ncols = self.maze.shape

            self.__ax2.clear()
            self.__ax2.set_xticks(np.arange(0.5, nrows, step=1))
            self.__ax2.set_xticklabels([])
            self.__ax2.set_yticks(np.arange(0.5, ncols, step=1))
            self.__ax2.set_yticklabels([])
            self.__ax2.grid(True)
            self.__ax2.plot(*self.__exit_cell, "gs", markersize=30)  # exit is a big green square
            self.__ax2.text(*self.__exit_cell, "Exit", ha="center", va="center", color="white")

            for cell in self.empty:
                q = model.q(cell) if model is not None else [0, 0, 0, 0]
                a = np.nonzero(q == np.max(q))[0]

                for action in a:
                    dx = 0
                    dy = 0
                    if action == Action.MOVE_LEFT:
                        dx = -0.2
                    if action == Action.MOVE_RIGHT:
                        dx = +0.2
                    if action == Action.MOVE_UP:
                        dy = -0.2
                    if action == Action.MOVE_DOWN:
                        dy = 0.2

                    # color (red to green) represents the certainty
                    color = clip((q[action] - -1) / (1 - -1))

                    self.__ax2.arrow(*cell, dx, dy, color=(1 - color, color, 0), head_width=0.2, head_length=0.1)

            self.__ax2.imshow(self.maze, cmap="binary")
            self.__ax2.get_figure().canvas.draw()

    def plot_environment(self, show_plot=True, folder=None, filename=None):
        nrows, ncols = self.maze.shape
        # plt.ion()
        if not show_plot: # don't show when show_plot is True
            plt.ioff()

        self.fig, self.ax3 = plt.subplots(1, 1, tight_layout=True, figsize=(5, 5))

        self.ax3.clear()
        self.ax3.set_xticks(np.arange(0, ncols, step=1))
        self.ax3.set_xticks(np.arange(0.5, ncols, step=1), minor=True)
        self.ax3.set_xticklabels(np.arange(0, ncols, 1))
        self.ax3.set_yticks(np.arange(0, nrows, step=1))
        self.ax3.set_yticks(np.arange(0.5, nrows, step=1), minor=True)
        self.ax3.set_yticklabels(np.arange(0, nrows, 1))

        # M = self.ax3.transData.get_matrix()
        # xscale = M[0, 0]
        # yscale = M[1, 1]
        # print(xscale, yscale)
        self.ax3.grid(True, 'minor')

        # marker_size = int(320/max(nrows,ncols)) # note that this 320 is empirically found for a fig of 5x5. The matplotlib docu sucks on the meaning of points
        # scatter = ax3.scatter(1,2, s=yscale**2, c=123, marker='s')
        # self.ax3.plot(self.__current_cell[0], self.__current_cell[1], "rs",
        #               markersize=marker_size)  # start is a big red square
        # self.ax3.text(self.__current_cell[0], self.__current_cell[1], "Start", ha="center", va="center", color="white")
        # # plot target:
        # self.ax3.plot(self.__exit_cell[0], self.__exit_cell[1], "gs",
        #               markersize=marker_size)  # exit is a big green square
        # self.ax3.text(self.__exit_cell[0], self.__exit_cell[1], "Exit", ha="center", va="center", color="white")
        # # plot obstacles:
        # self.ax3.imshow(self.maze, cmap="binary")
        # plt.ion()

        maze_visual = np.ones((nrows, ncols, 3))
        # plot target position:
        maze_visual[self.__exit_cell[1], self.__exit_cell[0], :] = [0,1,0]
        # self.ax3.text(self.__exit_cell[0], self.__exit_cell[1], "Exit", ha="center", va="center", color="black")
        maze_visual[self.__current_cell[1], self.__current_cell[0], :] = [1,0,0]
        # self.ax3.text(self.__current_cell[0], self.__current_cell[1], "Start", ha="center", va="center", color="black")
        obstacles = self.get_obstacle_positions()
        for obs in obstacles:
            maze_visual[obs[1], obs[0], :] = [0, 0, 0]

        self.ax3.imshow(maze_visual)

        self.ax3.get_figure().canvas.draw()
        self.ax3.get_figure().canvas.flush_events()
        # plt.pause(0.05)
        print('end')

        if filename:
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(folder+filename)
            plt.close('all')


    def get_exit_cell(self):
        return np.array([*self.__exit_cell])

    def get_flattend_position_vector(self, pos):
        mat = np.zeros((self.nrows, self.ncols), dtype=int)
        mat[pos[1], pos[0]] = 1
        # print(mat)
        vec = np.ndarray.flatten(mat)
        return vec

    def get_position_in_maze(self, pos):
        original_maze_format = np.reshape(pos, (self.nrows, self.ncols))
        org_pos = np.where(original_maze_format == 1)
        org_pos = (org_pos[1][0], org_pos[0][0])
        return org_pos

    def get_obstacle_positions(self):
        obstacle_indices = np.where(self.maze == 1)  # in 0 there are the y coordinates when we take the origin in the top left
        return np.asarray(list(zip(obstacle_indices[1], obstacle_indices[0])))  # list of obstacle positions

    def create_observation_vec(self):
        # maze_flattend = np.ndarray.flatten(self.maze)
        # exit = self.get_exit_cell()
        # target_position = self.get_flattend_position_vector(exit)
        # current_pos = tuple(self.__observe())
        # agent_pos = self.get_flattend_position_vector(current_pos)
        # observation_vec = np.concatenate((maze_flattend, target_position, agent_pos))
        # return observation_vec

        # new state vec: where all position are encoded as state:

        obstacle_pos = self.get_obstacle_positions().flatten()
        target_pos = self.get_exit_cell()
        agent_pos = self.__observe()
        if obstacle_pos.size == 0: # fix such that when there are no obstacles the output are still integer values
            observation_vec = np.concatenate((target_pos, agent_pos))
        else:
            observation_vec = np.concatenate((obstacle_pos, target_pos, agent_pos))
        return observation_vec.astype(np.float32)

