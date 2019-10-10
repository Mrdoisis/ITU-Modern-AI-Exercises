from pacman import *
from game import Agent

class AStarAgent(Agent):

    def getAction(self, state):
        best_path = self.getBestPath(state)
        return "Stop"


    def getBestPath(self, state):
        best_path = []
        start = state

        openList = []
        closedList = []
        openList.append((0, start, None))

        while not len(openList) < 1:
            q = min(openList, key=getFitnessFromTuple)
            s = getStateFromTuple(q)
            legal_actions = s.getLegalPacmanActions()
            successors = {s.generatePacmanSuccessor(action):action for action in legal_actions}
            for successor in successors:
                if successor.isWin():
                    best_path.append(successors[successor])
                    break
                g = getFitnessFromTuple(q) + 1 # step is 1
                food = successor.getFood()
                pos = successor.getPacmanPosition()
                #h =
                print "hi"


        return "Stop"


def getFitnessFromTuple(tup):
    return tup[0]

def getStateFromTuple(tup):
    return tup[1]

def getParentFromTuple(tup):
    return tup[2]

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

if __name__ == '__main__':
    # str_args = ['-l', 'tinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '100']
    # str_args = ['-l', 'testMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(sys.argv[1:])
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = AStarAgent()
    out = runGames(**args)

    scores = [o.state.getScore() for o in out]
    print(scores)