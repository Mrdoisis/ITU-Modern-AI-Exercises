import copy
import uuid
import numpy as np

from pacman import *
from game import Agent

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

if __name__ == '__main__':
    #str_args = ['-l', 'tinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '100']
    #str_args = ['-l', 'testMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(sys.argv[1:])
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = MCTSagent()
    out = runGames( **args)

    scores = [o.state.getScore() for o in out]
    print(scores)
