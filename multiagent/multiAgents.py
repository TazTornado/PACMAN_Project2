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
		some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

		initialScore = successorGameState.getScore()
		foodDists = []
		coefficient = 0
		for foodot in newFood.asList():
			foodDists.append(manhattanDistance(newPos, foodot))

		if len(foodDists) != 0:
			coefficient = 1/min(foodDists)
				
		return initialScore + coefficient

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

		gameState.isWin():
		Returns whether or not the game state is a winning state

		gameState.isLose():
		Returns whether or not the game state is a losing state
		"""
		"*** YOUR CODE HERE ***"
	
		def MiniMax(gameState, depth, agentIndex):
			"""
				Helper recursive function that executes the actual Minimax algorithm
			"""

			# base case: max depth or terminal state => return the score
			if depth == self.depth or gameState.isLose() or gameState.isWin():
				return self.evaluationFunction(gameState)

			legalMoves = gameState.getLegalActions(agentIndex)
			newDepth = depth            # depth value for next rec. call; may remain the same
			utilities = []

			# PACMAN AGENT => MAX_VALUE ALGORITHM
			if agentIndex == self.index:    
				# gather a list of successor states that correspond to the legal moves
				successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
				newIndex = agentIndex + 1   # prepare for next agent, i.e a ghost

				for newState in successorGameStates:
					utilities.append(MiniMax(newState, newDepth, newIndex))    # recursive call for next agent

				return max(utilities)
			
			# GHOST AGENT => MIN_VALUE ALGORITHM
			else:       # ghost agent => execute min
				newIndex = (agentIndex + 1) % gameState.getNumAgents()      # loop around the available agent indices
				if newIndex == self.index:      # if this is the last ghost
					newDepth = depth + 1        # then go 1 level deeper

				# gather a list of successor states that correspond to the legal moves
				successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
				
				for newState in successorGameStates:
					utilities.append(MiniMax(newState, newDepth, newIndex))    # recursive call for next agent
				
				return min(utilities)

################################################################################################################
		"""
			Perform the root's algorithm (pacman), for depth == 0 => root.    
			The root executes pretty much the same algorithm as the MiniMax  
			function. The difference is that root has to choose and return  
			the _action_ that corresponds to the max score returned, whereas 
			Minimax() only returns scores.
		"""                                   

		utilities = []
		legalMoves = gameState.getLegalActions(self.index)
		for move in legalMoves:
			possibleState = gameState.generateSuccessor(self.index, move)

			# begin recursive execution, starting from ghosts at depth 0
			utilities.append((move, MiniMax(possibleState, 0, self.index + 1))) 

		chosenPair = max(utilities, key = lambda item: item[1] )    # get the (move, value) pair with max value
		return chosenPair[0]

		util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
	"""
	Your minimax agent with alpha-beta pruning (question 3)
	"""

	def getAction(self, gameState):
		"""
		Returns the minimax action using self.depth and self.evaluationFunction
		"""
		"*** YOUR CODE HERE ***"

		def AlphaBeta(gameState, a, b, depth, agentIndex):
			"""
				Helper function that executes the maximizer and minimizer algorithms recursively.
				Returns a utility value.
			"""

			# base case: max depth or terminal state => return the score
			if depth == self.depth or gameState.isLose() or gameState.isWin():
				return self.evaluationFunction(gameState)

			legalMoves = gameState.getLegalActions(agentIndex)
			newDepth = depth            # depth value for next rec. call; may remain the same

			# PACMAN AGENT => MAXIMIZER ALGORITHM
			if agentIndex == self.index:
				v = float("-inf")		# initialize v to use when pruning
				newIndex = agentIndex + 1   # prepare for next agent, i.e a ghost

				for move in legalMoves:
					newState = gameState.generateSuccessor(agentIndex, move)
					utility = AlphaBeta(newState, a, b, newDepth, newIndex)    # recursive call for next agent
					v = max(v, utility)
					if v > b:       # pruning condition
						return v
					a = max(v, a)   # prepare for next newState

				return v

			# GHOST AGENT => MINIMIZER ALGORITHM
			else:
				v = float("inf")		# initialize v to use when pruning
				newIndex = (agentIndex + 1) % gameState.getNumAgents()      # loop around the available agent indices
				if newIndex == self.index:      # if this is the last ghost
					newDepth = depth + 1        # then go 1 level deeper

				for move in legalMoves:
					newState = gameState.generateSuccessor(agentIndex, move)
					utility = AlphaBeta(newState, a, b, newDepth, newIndex)
					v = min(v, utility)
					if v < a:		# pruning condition
						return v
					b = min(v, b)	# prepare for next newState

				return v
					
################################################################################################################
		"""
			Perform the root's algorithm (pacman), for depth == 0 => root    
			The root decides and returns the action corresponding to the best utility value
			returned from the helper functions.
		"""
		a = float("-inf"); b = float("inf") 	# initialize alpha and beta accordingly
		v = float("-inf")						# initialize v to use for pruning
		legalMoves = gameState.getLegalActions(self.index)

		# random assignment; it will change if it isn't the best action
		chosenMove = legalMoves[0]				

		for move in legalMoves:
			possibleState = gameState.generateSuccessor(self.index, move)

			# begin recursive execution, starting from ghosts at depth 0
			utility = AlphaBeta(possibleState, a, b, 0, self.index + 1)

			if v < utility:			# equivalent to v = max(v, utility)
				v = utility			# but now the action also has to be saved
				chosenMove = move
				
			if v > b:				# pruning condition
				return move
			a = max(v, a) 

		return chosenMove			# the best move found in iteration

		util.raiseNotDefined()

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

		def ExpectiMax(gameState, depth, agentIndex):
			"""
				Helper recursive function that executes the actual Minimax algorithm
			"""

			# base case: max depth or terminal state => return the score
			if depth == self.depth or gameState.isLose() or gameState.isWin():
				return self.evaluationFunction(gameState)

			legalMoves = gameState.getLegalActions(agentIndex)
			newDepth = depth            # depth value for next rec. call; may remain the same
			utilities = []

			# PACMAN AGENT => MAX_VALUE ALGORITHM
			if agentIndex == self.index:    
				# gather a list of successor states that correspond to the legal moves
				successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
				newIndex = agentIndex + 1   # prepare for next agent, i.e a ghost

				for newState in successorGameStates:
					utilities.append(ExpectiMax(newState, newDepth, newIndex))    # recursive call for next agent

				return max(utilities)
			
			# GHOST AGENT => MIN_VALUE ALGORITHM
			else:       # ghost agent => execute min

				#################################################################
				# It's assumed that ghost agents choose randomly(not optimally) #
				# one of their legal actions. Therefore each action's score 	#
				# will be multiplied by the probability of being chosen.		#
				# Uniform distribution => equal probability. Now each score		#
				# will actually be an aliquot part of the final score			#
				#################################################################
				pr = 1 / len(legalMoves)

				newIndex = (agentIndex + 1) % gameState.getNumAgents()      # loop around the available agent indices
				if newIndex == self.index:      # if this is the last ghost
					newDepth = depth + 1        # then go 1 level deeper

				# gather a list of successor states that correspond to the legal moves
				successorGameStates = [gameState.generateSuccessor(agentIndex, action) for action in legalMoves]
				
				for newState in successorGameStates:
					utilities.append(pr * ExpectiMax(newState, newDepth, newIndex))    # recursive call for next agent
				
				# calculate the score by adding up all the 
				return sum(utilities)

################################################################################################################
		"""
			Perform the root's algorithm (pacman), for depth == 0 => root.    
			The root executes pretty much the same algorithm as the MiniMax  
			function. The difference is that root has to choose and return  
			the _action_ that corresponds to the max score returned, whereas 
			Minimax() only returns scores.
		"""                                   

		utilities = []
		legalMoves = gameState.getLegalActions(self.index)
		
		for move in legalMoves:
			possibleState = gameState.generateSuccessor(self.index, move)

			# begin recursive execution, starting from ghosts at depth 0
			utilities.append((move, ExpectiMax(possibleState, 0, self.index + 1))) 

		chosenPair = max(utilities, key = lambda item: item[1] )    # get the (move, value) pair with max value
		return chosenPair[0]


		util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
	"""
	Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
	evaluation function (question 5).

	DESCRIPTION: <write something here so we know what you did>
	"""
	"*** YOUR CODE HERE ***"

	# actionScores = []
	# for action in currentGameState.getLegalActions():
	# 	successorGameState = currentGameState.generatePacmanSuccessor(action)
	# 	newPos = successorGameState.getPacmanPosition()
	# 	newFood = successorGameState.getFood()
	# 	initialScore = successorGameState.getScore()
	# 	foodDists = []
	# 	coefficient = 0
	# 	for foodot in newFood.asList():
	# 		foodDists.append(manhattanDistance(newPos, foodot))

	# 	if len(foodDists) != 0:
	# 		coefficient = 1/min(foodDists)

	# 	actionScores.append(initialScore + coefficient)

	# if actionScores:			
	# 	return max(actionScores)
	# else:
	# 	return 0

	util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
