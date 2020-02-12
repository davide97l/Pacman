# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import Directions
import random, util
import numpy as np

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            return max(self.minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(0, gameState))
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            return min(self.minimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(agent, gameState))

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):

    def alphabeta(self, agent, depth, gameState, alpha, beta):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            value = -999999
            for action in getLegalActionsNoStop(agent, gameState):
                value = max(value, self.alphabeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                alpha = max(alpha, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            for action in getLegalActionsNoStop(agent, gameState):
                value = 999999
                value = min(value, self.alphabeta(nextAgent, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                beta = min(beta, value)
                if beta <= alpha:  # alpha-beta pruning
                    break
            return value

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction using alpha-beta pruning.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]


class ExpectimaxAgent(MultiAgentSearchAgent):

    def expectimax(self, agent, depth, gameState):
        if gameState.isLose() or gameState.isWin() or depth == self.depth:
            return self.evaluationFunction(gameState)
        if agent == 0:  # maximize for pacman
            return max(self.expectimax(1, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(0, gameState))
        else:  # minimize for ghosts
            nextAgent = agent + 1  # get the next agent
            if gameState.getNumAgents() == nextAgent:
                nextAgent = 0
            if nextAgent == 0:  # increase depth every time all agents have moved
                depth += 1
            return sum(self.expectimax(nextAgent, depth, gameState.generateSuccessor(agent, action)) for action in
                       getLegalActionsNoStop(agent, gameState)) / float(len(getLegalActionsNoStop(agent, gameState)))

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.expectimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


def evaluationFunction(currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    """Calculate distance to the nearest food"""
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.manhattanDistance(newPos, food) for food in newFoodList]
    min_food_distance = 0
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]

    """Calculate the distance to nearest ghost"""
    ghostPositions = np.array(successorGameState.getGhostPositions())
    distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
    min_ghost_distance = 0
    nearestGhostScaredTime = 0
    if len(ghostPositions) > 0:
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
        # avoid certain death
        if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
            return -999999
        # eat a scared ghost
        if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    value = successorGameState.getScore() - min_food_distance
    if nearestGhostScaredTime > 0:
        # follow ghosts if scared
        value -= min_ghost_distance
    else:
        value += min_ghost_distance
    return value


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions
