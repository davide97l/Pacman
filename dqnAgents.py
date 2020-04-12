# Used code from
# DQN implementation by Tejas Kulkarni found at
# https://github.com/mrkulk/deepQN_tensorflow

# Used code from:
# The Pacman AI projects were developed at UC Berkeley found at
# http://ai.berkeley.edu/project_overview.html


import numpy as np
import random
import util
import time
import sys
import os

# Pacman game
from pacman import Directions
from game import Agent
import game

# Replay memory
from collections import deque

# Neural nets
from DQN import *

# only params value can be modified
params = {
    # Model backups
    'load_file': "model-trcikyClassic_585898_9422",  # relative path to the saved model
    'save_file': "smallClassic",  # name of the model
    'save_interval': 100000,  # Number of steps between each checkpoint

    # Training parameters
    'train_start': 5000,    # Steps before training starts
    'batch_size': 32,       # Replay memory batch size
    'mem_size': 100000,     # Replay memory size

    'discount': 0.95,       # Discount rate (gamma value)
    'lr': .0002,            # Learning reate

    # Epsilon value (epsilon-greedy)
    'eps': 1.0,             # Epsilon start value
    'eps_final': 0.1,       # Epsilon end value
    'eps_step': 10000       # Epsilon steps between start and end (linear)
}                     


class DQNAgent(game.Agent):
    def __init__(self, width, height, numTraining=0):

        # Load parameters from user-given arguments
        self.params = params
        self.params['width'] = width  # Maze width
        self.params['height'] = height  # Maze height
        self.params['num_training'] = numTraining  # Number of games used for training

        # create saves and logs directory
        if not os.path.exists("saves/DQN/"):
            os.makedirs("saves/DQN/")
        if not os.path.exists("logs/"):
            os.makedirs("logs/")

        # get saves directory
        if params["load_file"] is not None and not params["load_file"].startswith("saves/DQN/"):
            params["load_file"] = "saves/DQN/" + params["load_file"]

        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.qnet = DQN(self.params)  # create DQN

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.Q_global = []  # max Q-values in the current game
        self.cost_disp = 0   # current loss

        self.cnt = self.qnet.sess.run(self.qnet.global_step)  # number of steps the model has been trained so far
        self.local_cnt = 0  # number of total steps the algorithm has run

        self.numeps = 0  # current episode
        if params["load_file"] is not None:
            self.numeps = int(params["load_file"].split("_")[-1])
        self.last_score = 0  # Score in the last step
        self.s = time.time()  # time elapsed since beginning of training
        self.last_reward = 0.  # Reward obtained in the last step

        self.replay_mem = deque()  # replay memory used for training

        self.terminal = False  # True if the game in a terminal state

        self.last_score = 0  # Score obtained in the last state
        self.current_score = 0   # Score obtained in the current state
        self.last_reward = 0.  # Reward obtained in the last state
        self.ep_rew = 0  # Cumulative reward obtained in the current game

        self.last_state = None  # Last state
        self.current_state = None  # Current state
        self.last_action = None  # Last action
        self.won = True  # True if the game has been won
        self.delay = 0
        self.frame = 0

    # Select a move according to exploitation / exploration tradeoff
    def getMove(self):

        # Exploit / Explore
        if np.random.rand() >= self.params['eps']:
            # Exploit action
            self.Q_pred = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                                                     (1, self.params['width'], self.params['height'], 6)), 
                             self.qnet.q_t: np.zeros(1),
                             self.qnet.actions: np.zeros((1, 4)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]
            self.Q_global.append(max(self.Q_pred))
            a_winner = np.argwhere(self.Q_pred == np.amax(self.Q_pred))

            if len(a_winner) > 1:
                move = self.get_direction(
                    a_winner[np.random.randint(0, len(a_winner))][0])
            else:
                move = self.get_direction(
                    a_winner[0][0])
        else:
            # Random:
            move = self.get_direction(np.random.randint(0, 4))

        # Save last_action
        self.last_action = self.get_value(move)

        return move

    # converts direction to value
    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        else:
            return 3.

    # converts value to direction
    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        else:
            return Directions.WEST

    # make an observation on the reply memory, then use it to train the model on one batch
    def observation_step(self, state):
        if self.last_action is not None:
            # Process current experience state
            self.last_state = np.copy(self.current_state)
            # get the matrix of the new state
            self.current_state = self.getStateMatrices(state)

            # Process current experience reward
            self.current_score = state.getScore()
            # get the reward obtained in the current state
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            self.last_reward = reward  # experimental (don't change the reward)
            if reward <= -100:
                self.won = False
            self.ep_rew += self.last_reward

            # Store last experience into memory 
            experience = (self.last_state, float(self.last_reward), self.last_action, self.current_state, self.terminal)
            self.replay_mem.append(experience)
            if len(self.replay_mem) > self.params['mem_size']:
                self.replay_mem.popleft()

            # Save model
            if params['save_file']:
                if self.local_cnt > self.params['train_start'] and self.local_cnt % self.params['save_interval'] == 0:
                    self.qnet.save_ckpt('saves/DQN/model-' + params['save_file'] + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train
            self.train()

        # Next
        self.local_cnt += 1
        self.frame += 1
        self.params['eps'] = max(self.params['eps_final'],
                                 1.00 - float(self.cnt) / float(self.params['eps_step']))
        if self.numeps >= params['num_training']:
            params['eps'] = 0

    # Do an observation after each step (this method is called in the game.py file after each step)
    def observationFunction(self, state):
        self.terminal = False
        self.observation_step(state)
        return state

    # After each game print pacman statistics (this method is called in the game.py file when a game finishes)
    def final(self, state):
        # Total reward accumulated in an episode
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.params['width'])+'-m-' +
                        str(self.params['height'])+'-x-'+str(self.params['num_training'])+'.log', 'a')
        game_log = ("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f | Q: %10f | won: %r \n" %
                    (self.numeps, self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.params['eps'],
                     max(self.Q_global, default=float('nan')), self.won))
        log_file.write(game_log)
        sys.stdout.write(game_log)
        sys.stdout.flush()

    # Train the model sampling a batch from the replay memory
    def train(self):
        # The train process starts only if has passed a certain number of steps in order to fill the replay memory
        if self.local_cnt > self.params['train_start']:
            batch = random.sample(self.replay_mem, self.params['batch_size'])
            batch_s = [] # States (s)
            batch_r = [] # Rewards (r)
            batch_a = [] # Actions (a)
            batch_n = [] # Next states (s')
            batch_t = [] # Terminal state (t)
            for i in batch:
                batch_s.append(i[0])
                batch_r.append(i[1])
                batch_a.append(i[2])
                batch_n.append(i[3])
                batch_t.append(i[4])
            batch_s = np.array(batch_s)
            batch_r = np.array(batch_r)
            batch_a = self.get_onehot(np.array(batch_a))
            batch_n = np.array(batch_n)
            batch_t = np.array(batch_t)

            # return global step (number of training iterations on batches) and loss
            self.cnt, self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r)

    # one-hot encode action
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.params['batch_size'], 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot

    def getStateMatrices(self, state):
        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.params['width'], self.params['height']
        observation = np.zeros((6, height, width))

        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)

        observation = np.swapaxes(observation, 0, 2)

        return observation

    # Init the initial state of the agent (this method is called in the game.py file when a game starts)
    def registerInitialState(self, state):  # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.numeps += 1

        if self.numeps >= params['num_training']:
            params['eps'] = 0

    # Returns an action from the agent (this method is called in the game.py file when the agent has to select an action)
    def getAction(self, state):
        move = self.getMove()
        # Stop moving when not legal
        legal = state.getLegalActions(0)
        if move not in legal:
            move = random.choice(legal)

        return move
