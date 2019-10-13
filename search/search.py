# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import Queue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    import Queue

    visited = []
    remaining = Queue.Queue()
    remaining.put(problem.getStartState())

    visited.append(problem.getStartState())
    while not remaining.empty():
        current = remaining.get()

        for neighbour in problem.getSuccessors(current):
            if neighbour[0] not in visited:
                remaining.put(neighbour[0])
                visited.append(neighbour[0])
    return visited



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    import Queue

    # the start node is the initial pacman position
    start_node = problem.getStartState()

    # 'open' is a priority queue of unexplored nodes initialized with the start node
    # the priority is the combined cost and heuristic for nodes
    open = Queue.PriorityQueue()
    open.put((0, start_node))

    # 'closed' is a list of explored nodes
    closed = []

    # 'g' is a dictionary to keep track of stepcost to get to any state
    # g(s) = stepcost from start to parent + stepcost from parent to s
    g = {start_node: 0}

    # 'h' is a dictionary to keep track of the heuristic cost of a given state
    h = {start_node: heuristic(start_node, problem)}

    # 'f' is a dictionary to keep track of combined stepcost and heuristic of any given state
    # f(s) = g(s) + h(s)
    f = {start_node: g[start_node] + h[start_node]}

    # flag to define when we should stop exploring new nodes
    goal_is_found = False

    # triplet to contain the successor that leads us to the goal state
    # this variable will be assigned a new value once a goal state is found
    goal_node = start_node

    # 'nodes' is a dictionary to keep track of how we arrived at any given state
    # contains the action that brought us to this state and the state from which we took the action
    # (action, parent)
    nodes = {start_node: (None, None)}

    # continuously explores one node at a time, until there are no more unexplored nodes
    while not open.empty() and not goal_is_found:

        # get and remove the first node in the open list
        (_, q) = open.get()

        # mark the node as explored
        closed.append(q)

        # iterate over all connected nodes (successors) of the unexplored node
        for successor in problem.getSuccessors(q):
            (s, action, stepCost) = successor

            # if we found a goal state then break
            if problem.isGoalState(s):
                nodes[s] = (action, q)
                goal_node = successor
                goal_is_found = True
                break

            # calculate combined cost of going to this successor state
            g_ = g[q] + stepCost
            h_ = heuristic(s, problem)
            print "("+action+","+str(h_)+")"
            f_ = g_ + h_

            # if we found a lower combined cost to a state that we already explored, then update the cost of that state
            if s in closed:
                if f_ < f[s]:
                    g[s] = g_
                    h[s] = h_
                    f[s] = f_
                    nodes[s] = (action, q)
            else:
                # queue the successor state for exploration
                g[s] = g_
                h[s] = h_
                f[s] = f_
                open.put((h[s], s))
                nodes[s] = (action, q)

    # backtrack from the goal state to the start state to get the solution
    (goal_action, goal_parent) = nodes[goal_node[0]]
    solution = [goal_action]
    parent = goal_parent
    done = False
    while not done:
        # insert action to the solution
        solution.insert(0, nodes[parent][0])

        # update parent to be parent's parent
        parent = nodes[parent][1]
        if nodes[parent][0] is None:
            done = True


    return solution




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
