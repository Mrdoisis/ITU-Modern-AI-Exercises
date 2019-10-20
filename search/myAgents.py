from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

import copy
import numpy as np


# Agent using Monte-Carlo-Tree-Search
class Node():
    def __init__(self, id, parent, children, action_taken, state):
        self.ID = id
        self.Parent = parent
        self.Children = children # dictionary of actions
        self.Value = 0
        self.Visits = 0
        self.Action = action_taken
        self.State = state
class MCTSagent(Agent):
    def __init__(self):
        self.explored = {}  # Dictionary for storing the explored states
        self.n = 5  # Depth of search  # TODO: Play with this once the code runs
        #self.c = 1/np.sqrt(2)  # Exploration parameter # TODO: Play with this once the code runs
        self.c = 0.2
        self.treeSize = 1
        self.explored = {}

    def getAction(self, state):
        """ Main function for the Monte Carlo Tree Search. For as long as there
            are resources, run the main loop. Once the resources runs out, take the
            action that looks the best.
        """
        root = Node(0, None, {}, None, state)
        for _ in range(self.n):
            v1 = self.tree_policy(root)
            result = self.defaultPolicy(v1.State)
            self.backup(v1, result)
        return self.best_child(root)

    def best_child(self, node):
        """ Given a state, return the best action according to the UCT criterion."""
        """ YOUR CODE HERE!"""
        actions = {}
        for child in node.Children.values():
            x_j = child.Value
            if x_j is None:
                x_j = 0
            actions[child.Action] = x_j + 2 * self.c * np.sqrt(2 * np.log(self.treeSize) / child.Visits) # UCT
        max_actions = [k for k, v in actions.iteritems() if v == max(actions.values())]
        legal_actions = []
        valid_actions = node.State.getLegalPacmanActions()
        for a in max_actions:
            if a in valid_actions:
                legal_actions.append(a)

        return np.random.choice(legal_actions)  # return random legal max action

    def tree_policy(self, node):
        while not node.State.isWin() and not node.State.isLose():
            if len(node.State.getLegalPacmanActions())-1 > len(node.Children):
                return self.expand(node)
            else:
                best_child = self.best_child(node)
                node = node.Children[best_child]
        return node

    def expand(self, node):
        untried_actions = node.State.getLegalPacmanActions()
        untried_actions.remove("Stop")
        for tried_action in node.Children:
            untried_actions.remove(tried_action)
        chosen_untried_action = np.random.choice(untried_actions)
        state_when_action_taken = node.State.generatePacmanSuccessor(chosen_untried_action)
        new_node = Node(len(self.explored), node, {}, chosen_untried_action, state_when_action_taken)
        node.Children[chosen_untried_action] = new_node
        return new_node

    def defaultPolicy(self, state):
        s = state
        while not s.isWin() and not s.isLose():
            legal_actions = s.getLegalPacmanActions()
            legal_actions.remove("Stop")
            rnd_action = np.random.choice(legal_actions)
            s = s.generatePacmanSuccessor(rnd_action)
        return s.getScore()

    def backup(self, node, value):
        n = node
        while not n is None:
            n.Visits += 1
            n.Value = n.Value + value
            n = n.Parent

# Agent using Q-learning
class QLearningAgent(Agent):
    """
    This controller is inspired by the approach taken in the Q-learning videos and blog posts tutorials by https://deeplizard.com.
    """

    def getActionValues(self, state):
        # Returns the Q-values for all actions in a given state.
        # If it is the first time we see the state, then add a new entry with all q-values initially 0
        if not state in self.q_values:
            self.q_values[state] = [0, 0, 0, 0, 0]

            # Set the Q-value of all illegal values to -infinity
            for a in self.actions:
                if a not in state.getLegalPacmanActions():
                    i = self.actions.index(a)
                    self.q_values[state][i] = -np.infty
        return self.q_values[state]

    def registerInitialState(self, state_org):
        """
        This agent will do all its training every time it is initialized.
        Once the training is done, the agent will exploit the learned Q-table in the getAction function.
        """

        self.actions = ["North", "East", "South", "West", "Stop"]
        self.action_space_size = len(self.actions)

        # Q-Table with all Q-values initially 0.
        self.q_values = {}

        # Q-learning parameters
        self.num_episodes = 1000
        self.learning_rate = 0.1
        self.discount_rate = 0.99

        # Epsilon-greedy parameters used for trade off between exploration-exploitation
        # The epsilon value (exploration_rate) will decay after each episode such that it will be
        # more and more likely to exploit rather than explore
        self.exploration_rate  = 1
        self.max_exploration_rate = 1
        self.min_exploration_rate = 0.01
        self.exploration_decay_rate = 0.001

        # Q-learning algorithm
        rewards_all_episodes = []
        init_state = state_org

        for episode in range(self.num_episodes):
            state = init_state
            done = False
            rewards_current_episode = 0

            while(True):

                # Exploration-exploitation trade-off
                # epsilon-greedy to decide whether to explore or exploit
                exploration_rate_threshold = np.random.uniform(0, 1)
                if exploration_rate_threshold > self.exploration_rate:
                    # exploit
                    # If multiple actions have the same Q-value in this state, then choose one of them at random
                    max_actions = [i for i, x in enumerate(self.getActionValues(state)) if x == max(self.getActionValues(state))]
                    action = self.actions[np.random.choice(max_actions)]
                else:
                    # explore
                    legal = state.getLegalPacmanActions()
                    action = np.random.choice(legal)

                # Take new action
                new_state = state.generatePacmanSuccessor(action)
                reward = new_state.getScore() - state.getScore()
                if new_state.isWin() or new_state.isLose():
                    done = True

                action_index = self.actions.index(action)

                # Update Q-table
                self.getActionValues(state)[action_index] = self.getActionValues(state)[action_index] * (1 - self.learning_rate) + \
                    self.learning_rate * (reward + self.discount_rate * np.nanmax(self.getActionValues(new_state)))

                # Transition to the next state
                state = new_state

                # Add new reward
                rewards_current_episode += reward

                if done: break

            # Decay exploration rate proportional to its current value using exponential decay
            self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)

            # Add current episode reward to total rewards list
            print "Episode " + str(episode) + ": " + str(state.getScore())
            rewards_all_episodes.append(rewards_current_episode)

        # Calculate and print the average reward per hundred episodes
        rewards_per_hundred_episodes = np.split(np.array(rewards_all_episodes), self.num_episodes / 100)
        count = 100

        print("********Average reward per hundred episodes********\n")
        for r in rewards_per_hundred_episodes:
            print str(count) + ": " + str(sum(r / 100))
            count += 100



    def getAction(self, state):
        # Exploit the Q-Table in order to choose the best expected action in current state.
        max_actions = [i for i, x in enumerate(self.getActionValues(state)) if x == max(self.getActionValues(state))]
        action = self.actions[np.random.choice(max_actions)]
        return action


# Agent using A* and foodHeuristic
class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='aStarSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP
class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost
class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem
def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    "*** YOUR CODE HERE ***"
    foodList = foodGrid.asList()

    if len(foodList) == 0: return 0

    # calculate manhattan distance to all foods
    distances = []
    for food in foodList:
        distances.append(util.manhattanDistance(position, food))

    # point to the closest food
    closest = foodList[distances.index(min(distances))]

    #
    remaining_foods = 0
    for (x, y) in foodList:
        if (x != position[0] and x != closest[0]) or (y != position[1] and y != closest[1]):
            remaining_foods = remaining_foods + 1

    real_dist_to_closest = mazeDistance(position, closest, problem.startingGameState)
    return real_dist_to_closest + remaining_foods
def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))