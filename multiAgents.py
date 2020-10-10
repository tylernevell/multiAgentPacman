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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # everytime we call out eval func, we need to evaluate our current position
        
        # food locations built into list to iterate through, using a class function asList()
        newFoodList = newFood.asList()
        # print(action)
        if action == "Stop":
            # print(action)
            return float("-inf")
        
        minDistToFood = float("inf")
        # first, we need to see where food is around us
        for foodPellet in newFoodList:
            # distance to food pellet from our current position
            distToFood = manhattanDistance(newPos, foodPellet)
            #print(distToFood)
            
            if distToFood <= minDistToFood:
                minDistToFood = distToFood
            
        
        minDistToFood_score = (1.0 / minDistToFood)

        # need to get distance from ghost
        distToGhost = 1
        for ghostState in newGhostStates:
            #print(ghost.getPosition())
            distToGhost += manhattanDistance(newPos, ghostState.getPosition())
        
        distToGhost_score = (1.0 / distToGhost)
        # print(minDistToFood)
        # print("-----------")
        print(successorGameState.getScore())
        return successorGameState.getScore() + minDistToFood_score - distToGhost_score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        # score the leaves with self.evaluationFunction
        
        # returning only the values... need to return the action

        def miniMaxValue(gameState, depth, agentIndex):
            # player <- game.TO-MOVE(state)
            # value, move <- MAX-VALUE(game,state)
            # return move

            # searching new tree of possible actions at our lowest possible depth. 
            if gameState.getNumAgents() == agentIndex:
                agentIndex = 0
                depth += 1
            

            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), "Stop")
            if depth >= self.depth:
                return (self.evaluationFunction(gameState), "Stop")
            # Pacman agent number = 0, so we look for a maxValue
            if agentIndex == 0:
                return maxValue(gameState, depth, agentIndex)
            # if not zero, then probs ghost, which is looking for minvalue, so lets get that
            else:
                return minValue(gameState, depth, agentIndex)



        def maxValue(gameState, depth, agentIndex):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- -inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MIN-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = float("-inf")
            move = "Stop"
            max_tuple = (v, "")

            pacmanActions = gameState.getLegalActions(agentIndex)

            if not pacmanActions:
                return (self.evaluationFunction(gameState), "Stop")

            for action in pacmanActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                temp_tuple = miniMaxValue(successor, depth, agentIndex + 1)
                v = temp_tuple[0]
                if v > max_tuple[0]:
                    max_tuple = (v, action)

            return max_tuple

        def minValue(gameState, depth, agentIndex):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- +inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = float("inf")
            move = "Stop"
            min_tuple = (v, "")

            ghostActions = gameState.getLegalActions(agentIndex)

            if not ghostActions:
                return (self.evaluationFunction(gameState), "Stop")

            for action in ghostActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                temp_tuple = miniMaxValue(successor, depth, agentIndex + 1)
                v = temp_tuple[0]
                if v < min_tuple[0]:
                    min_tuple = (v, action)

            return min_tuple

        return miniMaxValue(gameState, 0, 0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def miniMaxValue(gameState, depth, agentIndex, alpha, beta):
            # player <- game.TO-MOVE(state)
            # value, move <- MAX-VALUE(game,state)
            # return move

            # We've conducted actions for every agent
            if gameState.getNumAgents() == agentIndex:
                
                # need to reset back to pacman and let him move while
                agentIndex = 0
                
                # exploring the next level in our tree of possible actions 
                depth += 1
            
            # I've ensured every return value is a tuple in order to prevent errors and be able to
            # keep track of the move that comes with our optimal values

            # we've won
            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), "Stop")
            
            # we're out of our element as an agent and need to let the algorithm catch up
            if depth >= self.depth:
                return (self.evaluationFunction(gameState), "Stop")
            
            # Pacman agent number = 0, so we look for a maxValue
            if agentIndex == 0:
                return maxValue(gameState, depth, agentIndex, alpha, beta)
            
            # if not zero, then probs ghost, which is looking for minvalue, so lets get that
            else:
                return minValue(gameState, depth, agentIndex, alpha, beta)



        def maxValue(gameState, depth, agentIndex, alpha, beta):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- -inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MIN-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = float("-inf")
            move = "Stop"
            max_tuple = (v, "")

            # where can pacman move?
            pacmanActions = gameState.getLegalActions(agentIndex)

            # no possible actions for the pac, so eval gamestate and stop movement
            if not pacmanActions:
                return (self.evaluationFunction(gameState), "Stop")

            # evaluate every legal action for pacman
            for action in pacmanActions:
                
                # what will our successor game state be? 
                successor = gameState.generateSuccessor(agentIndex, action)

                # recursively returns move and value of that move for this action's successor game state
                temp_tuple = miniMaxValue(successor, depth, agentIndex + 1, alpha, beta)
                
                # grab the return value of the successor game state
                v = temp_tuple[0]
                
                # is the returned value more advantageous than the current best outcome?
                if v > max_tuple[0]:

                    # if so, this is our new optimal move for pacman
                    max_tuple = (v, action)
                
                # direct from class pseudocode
                # but don't match on equality according to instructions
                if v > beta:
                    return (v, action)
                alpha = max(alpha, v)

            return max_tuple

        def minValue(gameState, depth, agentIndex, alpha, beta):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- +inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = float("inf")
            move = "Stop"
            min_tuple = (v, "")

            # where can ghost move?
            ghostActions = gameState.getLegalActions(agentIndex)

            # no possible actions for ghostie, so eval gamestate and stop movement
            if not ghostActions:
                return (self.evaluationFunction(gameState), "Stop")

            # evaluate every legal action for ghostie   
            for action in ghostActions:

                # what will our successor game state be? 
                successor = gameState.generateSuccessor(agentIndex, action)

                # recursively returns move and value of that move for this action's successor game state
                temp_tuple = miniMaxValue(successor, depth, agentIndex + 1, alpha, beta)
                
                # grab the return value of the successor game state
                v = temp_tuple[0]

                # is the returned value more advantageous than the current best outcome?
                if v < min_tuple[0]:
                    
                    # if so, this is our new optimal move for pacman
                    min_tuple = (v, action)
                
                # ripped from class pseudocode
                # but don't match on equality according to instructions
                if v < alpha:
                    return (v, action)
                beta = min(beta, v)

            return min_tuple

        return miniMaxValue(gameState, 0, 0, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimaxValue(gameState, depth, agentIndex):
            # player <- game.TO-MOVE(state)
            # value, move <- MAX-VALUE(game,state)
            # return move

            # searching new tree of possible actions at our lowest possible depth. 
            if gameState.getNumAgents() == agentIndex:
                agentIndex = 0
                depth += 1
            

            if gameState.isWin() or gameState.isLose():
                return (self.evaluationFunction(gameState), "Stop")
            if depth >= self.depth:
                return (self.evaluationFunction(gameState), "Stop")
            # Pacman agent number = 0, so we look for a maxValue
            if agentIndex == 0:
                return maxValue(gameState, depth, agentIndex)
            # if not zero, then probs ghost, which is looking for minvalue, so lets get that
            else:
                return expValue(gameState, depth, agentIndex)



        def maxValue(gameState, depth, agentIndex):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- -inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MIN-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = float("-inf")
            move = "Stop"
            max_tuple = (v, "")

            pacmanActions = gameState.getLegalActions(agentIndex)

            if not pacmanActions:
                return (self.evaluationFunction(gameState), "Stop")

            for action in pacmanActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                temp_tuple = expectimaxValue(successor, depth, agentIndex + 1)
                v = temp_tuple[0]
                if v > max_tuple[0]:
                    max_tuple = (v, action)

            return max_tuple

        def expValue(gameState, depth, agentIndex):
            # if game.IS-TERMINAL(state) then return game.UTILITY(state, player), null
            # v <- +inf
            # for each a in game.ACTIONS(state) do 
            #     v2, a2 <- MAX-VALUE(game, game.RESULT(state, a))
            #     if v2 > v then
            #         v, move <- v2, a
            # return v, move

            v = 0.0
            move = "Stop"
            exp_tuple = (v, "")

            ghostActions = gameState.getLegalActions(agentIndex)

            # probability of going down each branch
            p = 1.0 / len(ghostActions)

            if not ghostActions:
                return (self.evaluationFunction(gameState), "Stop")

            for action in ghostActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                temp_tuple = expectimaxValue(successor, depth, agentIndex + 1)
                v = temp_tuple[0]
                exp_tuple = (exp_tuple[0] + v * p, action)


            return exp_tuple

        return expectimaxValue(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pac_pos = currentGameState.getPacmanPosition()
    ghost_pos = currentGameState.getGhostPositions()


# Abbreviation
better = betterEvaluationFunction
