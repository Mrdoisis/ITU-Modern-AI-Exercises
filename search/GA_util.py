import numpy as np


opposites = {
    "North" : "South",
    "South": "North",
    "West": "East",
    "East": "West",
}


class Sequence:
    """ Continues until one failure is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        action = None
        for child in self.children:
            action = child(state)
            if not action:
                return False
        return action


class Selector:
    """ Continues until one success is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        for child in self.children:
            action = child(state)
            if action:
                return action
        return False


class CheckValid:
    """ Check whether <direction> is a valid action for PacMan
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        if self.direction in state.getLegalPacmanActions():
            return True
        return False


class CheckDanger:
    """ Check whether there is a ghost in <direction>, or any of the adjacent fields.
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        return self.is_dangerous(state)

    def is_dangerous(self, state):
        """ YOUR CODE HERE!"""
        new_state = state.generatePacmanSuccessor(self.direction)
        new_pacman_pos = new_state.getPacmanPosition()
        new_ghost_pos = new_state.getGhostPositions()

        if new_pacman_pos in new_ghost_pos:
            print "Oh jolly! I'm being chased from " + self.direction
            return True

        adj = new_state.getLegalPacmanActions()
        for legal_action in adj:
            future_state = new_state.generatePacmanSuccessor(legal_action)
            future_pacman = future_state.getPacmanPosition()
            future_ghosts = future_state.getGhostPositions()

            if future_pacman in future_ghosts:
                print "Oh no! There's a ghost in " + legal_action
                return True
        return False

class ActionGo:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self, direction="Random"):
        self.direction = direction

    def __call__(self, state):
        if self.direction == "Random":
            legal_actions = state.getLegalPacmanActions()
            legal_actions.remove("Stop")
            return np.random.choice(legal_actions)
        if self.direction in state.getLegalPacmanActions():
            return self.direction

class ActionGoNot:
    """ Go in a random direction that isn't <direction>
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        legal_actions = state.getLegalPacmanActions()
        if self.direction in legal_actions:
            legal_actions.remove(self.direction)
        return np.random.choice(legal_actions)


class DecoratorInvert:
    def __call__(self, arg):
        return not arg

def parse_node(genome, parent=None):
    if len(genome) == 0:
        return

    if isinstance(genome[0], list):
        parse_node(genome[0], parent)
        parse_node(genome[1:], parent)

    elif genome[0] == "SEQ":
        if parent is not None:
            node = parent.add_child(Sequence(parent))
        else:
            node = Sequence(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0] == 'SEL':
        if parent is not None:
            node = parent.add_child(Selector(parent))
        else:
            node = Selector(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0].startswith("Valid"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckValid(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Danger"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckDanger(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("GoNot"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGoNot(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Go"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGo(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0] == ("Invert"):
        arg = genome[0].split('.')[-1]
        parent.add_child(DecoratorInvert(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    else:
        print("Unrecognized in ")
        raise Exception

    return parent




