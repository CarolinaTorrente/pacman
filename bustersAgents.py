from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import numpy as np
import os.path

class NullGraphics(object):
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__( self, index = 0, inference = "ExactInference", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        #for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        #self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index = 0, inference = "KeyboardInference", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''
class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''
    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''
    def printGrid(self, gameState):
        table = ""
        #print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print( gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())


    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0) ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if   ( move_random == 0 ) and Directions.WEST in legal:  move = Directions.WEST
        if   ( move_random == 1 ) and Directions.EAST in legal: move = Directions.EAST
        if   ( move_random == 2 ) and Directions.NORTH in legal:   move = Directions.NORTH
        if   ( move_random == 3 ) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"

    def state_tick(self, gameState):  # contains information about the current state of the game
        # We will store the information we want in a variable and then the function will just return it
        # the first is x and the second is y
        pacman_pos = gameState.getPacmanPosition()
        ghost_pos = gameState.getGhostPositions()
        nearest_food_dist = gameState.getDistanceNearestFood()
        remaining_food = gameState.getNumFood()
        current_score = gameState.getScore()

        return (pacman_pos[0], pacman_pos[1], ghost_pos[0][0], ghost_pos[0][1], ghost_pos[1][0], ghost_pos[1][1],
                nearest_food_dist, remaining_food, current_score)

    def action(self, gameState): #is the action executed by Pacman in that state
        return self.chooseAction(gameState)

    def state_next_tick(self, gameState): # is the state reached after applying action in state tick
        return self.state_tick(gameState)

"""    def reward(self, gameState):  #is the reward obtained when performing the transition
        reward = 0
        if gameState.getNumFood() < self.last_food_count:
            # Reward for eating a food pellet
            reward += 1
        else:
            # Penalty for taking a step
            reward -= 1

        if gameState.isWin():
            # Big reward for winning the game
            reward += 1000

        # Update the last food count
        self.last_food_count = gameState.getNumFood()

        return reward"""


class QLearningAgent(BustersAgent):

    #Initialization
    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.epsilon = 0
        self.alpha = 0.8
        self.discount = 0.6
        self.actions = {"North":0, "East":1, "South":2, "West":3}
        if os.path.exists("qtable.txt"):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            #"*** CHECK: NUMBER OF ROWS IN QTABLE DEPENDS ON THE NUMBER OF STATES ***"
            self.initializeQtable(27)
        self.tuple = ()

    def initializeQtable(self, nrows):
        "Initialize qtable"
        self.q_table = np.zeros((nrows,len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)

        return q_table


    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()

        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item)+" ")
            self.table_file.write("\n")

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")


    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()


    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        """
        direction = self.direction(state)
        ghost_distance = self.ghost_distance(state)
        if ghost_distance < 3:
            row=0
            row += direction
        elif  3<=ghost_distance < 6:
            row=5
            row += direction
        elif 6 <= ghost_distance < 10:
            row=9
            row+= direction
        elif 6 <= ghost_distance < 10:
            row=9
            row+= direction
        elif 10 <= ghost_distance<14:
            row= 13
            row+= direction
        elif 14 <= ghost_distance < 16:
            row = 18
            row += direction
        elif 16 <= ghost_distance < 20:
            row = 23
            row += direction

        return row


    def ghost_distance(self, GameState):
        pos = GameState.getPacmanPosition()
        ghosts = GameState.getGhostPositions()
        num_ghosts = len(ghosts)

        # Get the closest ghost and its distance
        if num_ghosts > 0:
            min_distance = 20
            for i in range(num_ghosts):
                distance = self.distancer.getDistance(pos, ghosts[i])
                if distance < min_distance:
                    min_distance = distance
                    closest_ghost = ghosts[i]
            ghost_distance = min_distance
        else:
            ghost_distance = 20

        return ghost_distance

    def nearestGhost(self, gameState):
        ghosts_pos = gameState.getGhostPositions()
        pacman_pos = gameState.getPacmanPosition()

        distance = abs(ghosts_pos[0][0] - pacman_pos[0]) + abs(
                ghosts_pos[0][1] - pacman_pos[1])
        index = 0
        # Here we just append the distance of each ghost with respect to pacman
        for i in range(len(ghosts_pos)):
            dist = abs(ghosts_pos[i][0] - pacman_pos[0]) + abs(ghosts_pos[i][1] - pacman_pos[1])
            if dist < distance:
                distance = dist
                index = i

        return ghosts_pos[index], distance

    def direction(self, gameState):
        ghosts_pos, aux = self.nearestGhost(gameState)
        pacman_pos = gameState.getPacmanPosition()
        direction = 0
        legalActions = gameState.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if ghosts_pos[0] > pacman_pos[0] and "East" in legalActions:
            direction = 1
        elif ghosts_pos[1] > pacman_pos[1] and "North" in legalActions:
            direction = 0
        elif ghosts_pos[0] < pacman_pos[0] and "West" in legalActions:
            direction= 3
        elif ghosts_pos[1] < pacman_pos[1] and "South" in legalActions:
            direction =2

        # in the following section we check the existence of walls in the path
        elif len(legalActions) > 0:
            if ghosts_pos[1] > pacman_pos[1] and "North" not in legalActions:
                # pacman can't go to the north so we check west and east:
                if ghosts_pos[0] < pacman_pos[0] and "West" not in legalActions:
                    if "East" in legalActions:
                        direction=1
                    elif "East" not in legalActions:
                        # remaining is south
                        direction=2
                elif ghosts_pos[0] > pacman_pos[0] and "East" not in legalActions:
                    if "West" in legalActions:
                        direction = 3
                    elif "West" not in legalActions:
                        direction = 2

            elif ghosts_pos[1] < pacman_pos[1] and "South" not in legalActions:
                if ghosts_pos[0] < pacman_pos[0] and "West" not in legalActions:
                    if "East" not in legalActions:
                        # then go North
                        direction = 0
                    elif "East" in legalActions:
                        direction=1
                elif ghosts_pos[0] > pacman_pos[0] and "East" not in legalActions:
                    if "West" not in legalActions:
                        # then north as it's the remaining
                        direction=0
                    elif "West" in legalActions:
                        direction= 3

            elif ghosts_pos[0] < pacman_pos[0] and "West" not in legalActions:
                if ghosts_pos[1] > pacman_pos[1] and "North" not in legalActions:
                    if "East" in legalActions:
                        direction = 1
                    direction= 2
                elif ghosts_pos[1] < pacman_pos[1] and "South" not in legalActions:
                    if "North" in legalActions:
                        direction=0
                    direction=1

            elif ghosts_pos[0] > pacman_pos[0] and "East" not in legalActions:
                if ghosts_pos[1] > pacman_pos[1] and "North" not in legalActions:
                    if "West" in legalActions:
                        direction = 3
                    elif "West" not in legalActions:
                        direction=2
                elif ghosts_pos[1] < pacman_pos[1] and "South" not in legalActions:
                    if "North" in legalActions:
                        direction=0
                    direction=3

            #direction = self.actions[random.choice(legalActions)]
        return direction



    def getQValue(self, state, action):

        """
            Returns Q(state,action)
            Should return 0.0 if we have never seen a state
            or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]


    def computeValueFromQValues(self, state):
        """
            Returns max_action Q(state,action)
            where the max is over legal actions.  Note that if
            there are no legal actions, which is the case at the
            terminal state, you should return a value of 0.0.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, state):
        """
            Compute the best action to take in a state.  Note that if there
            are no legal actions, which is the case at the terminal state,
            you should return None.
        """
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        if len(legalActions)==0:
            return None

        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """

        # Pick Action
        legalActions = state.getLegalPacmanActions()
        if 'Stop' in legalActions: legalActions.remove("Stop")
        action = None

        if len(legalActions) == 0:
                return action

        flip = util.flipCoin(self.epsilon)

        if flip:
            return random.choice(legalActions)
        return self.getPolicy(state)


    def update(self, state, action, nextState, reward):
        """
            The parent class calls this to observe a
            state = action => nextState and reward transition.
            You should do your Q-Value update here

        Q-Learning update:

        if terminal_state:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
        else:
        Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # we take positions of the state and next state
        position = self.computePosition(state)
        next_position = self.computePosition(nextState)
        # compute action
        action_col = self.actions[action]

        # Update Q-value using Q-learning update

        if state.isWin():
            # If state is a win, we set the target value to the reward
            self.q_table[position][action_col] = (1 - self.alpha) * self.q_table[position][action_col] \
                                                 + self.alpha * (reward + 2)
        else:
            # we estimate the target value using the Q-value of the best action in the nextState
            max_q_next = max(self.q_table[next_position]) # best action for the next state

            # we apply the formula for qlearning update
            self.q_table[position][action_col] = (1 - self.alpha) * self.q_table[position][action_col] \
                                                 + self.alpha * (reward + self.discount * max_q_next)


    def getPolicy(self, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def getReward(self, state, action, nextstate):
        "Return the obtained reward"

        a = self.ghost_distance(state)
        b = self.ghost_distance(nextstate)

        # First the case were current state is positive
        pos = 1
        neg = -2
        if b<= a:
            if b < 3:
                reward = pos*(1)
            elif 3 <= b < 6:
                reward = pos*(0.95)
            elif 6 <= b< 10:
                reward = pos*(0.9)
            elif 6 <= b < 10:
                reward = pos*(0.85)
            elif 10 <= b < 14:
                reward = pos*(0.8)
            elif 14 <= b < 16:
                reward = pos*(0.75)
            elif 16 <= b< 20:
                reward = pos*(0.7)
        elif b > a:
            reward = neg

        return reward
        #util.raiseNotDefined()






