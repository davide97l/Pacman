from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np


class SearchAgent(Agent):

    def __init__(self, evalFn = 'evaluationFunction', depth='6'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class DeepSearchAgent(SearchAgent):

    def deepSearch(self, depth, gameState):
        if len(getLegalActionsNoStop(0, gameState)) == 0 or gameState.isLose() \
                or depth == self.depth or gameState.isWin():
            return self.evaluationFunction(gameState) - depth * 100
        newPos = gameState.getPacmanPosition()
        gameState.data.layout.walls[newPos[0]][newPos[1]] = True
        val = []
        for action in getLegalActionsNoStop(0, gameState):
            val.append(self.deepSearch(depth + 1, gameState.generateSuccessor(0, action)))
        max_val = max(val)
        gameState.data.layout.walls[newPos[0]][newPos[1]] = False
        return max_val + self.evaluationFunction(gameState) - depth * 100

    def getAction(self, gameState):

        # as the food begins running out it is better to decrease the research depth, in an empty maze there is not too much to search
        if gameState.getNumFood() <= self.depth:
            self.depth = gameState.getNumFood() - 1
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = []
        newPos = gameState.getPacmanPosition()
        gameState.data.layout.walls[newPos[0]][newPos[1]] = True
        for action in possibleActions:
            action_scores.append(self.deepSearch(0, gameState.generateSuccessor(0, action)))
        gameState.data.layout.walls[newPos[0]][newPos[1]] = False
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)

        """print(possibleActions)
        print(action_scores)
        input()"""

        return possibleActions[chosenIndex]


def evaluationFunction(currentGameState):

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    """Calculate distance to the nearest food"""
    newFoodList = np.array(newFood.asList())
    distanceToFood = [util.euclideanDistance(newPos, food) for food in newFoodList]
    min_food_distance = 0
    if len(newFoodList) > 0:
        min_food_distance = distanceToFood[np.argmin(distanceToFood)]

    """Calculate the distance to nearest ghost"""
    ghostPositions = np.array(currentGameState.getGhostPositions())
    if len(ghostPositions) > 0:
        distanceToGhost = [util.manhattanDistance(newPos, ghost) for ghost in ghostPositions]
        min_ghost_distance = distanceToGhost[np.argmin(distanceToGhost)]
        nearestGhostScaredTime = newScaredTimes[np.argmin(distanceToGhost)]
        # avoid certain death
        if min_ghost_distance <= 1 and nearestGhostScaredTime == 0:
            return -999999
        # eat a scared ghost
        if min_ghost_distance <= 1 and nearestGhostScaredTime > 0:
            return 999999

    return currentGameState.getScore() * 5 - min_food_distance


def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions

