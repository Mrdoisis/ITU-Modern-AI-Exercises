import numpy as np
import copy

import textDisplay

from pacman import *
from searchAgents import GAAgent

class EvolvePacManBT():
    def __init__(self, args, pop_size, num_parents, numGames=5):
        args['numGames'] = numGames
        # args['numTraining'] = args['numGames'] ## DOESN'T WORK # suppress the output
        self.display_graphics = args['display']
        args['display'] = textDisplay.NullGraphics()

        self.args = args

        self.pop_size = pop_size
        self.num_parents = num_parents
        self.gene_pool = None

        self.__create_initial_pop()

    def __create_initial_pop(self):
        self.gene_pool = [GAAgent()]
        self.produce_next_generation(self.gene_pool)

    def produce_next_generation(self, parents):
        """ YOUR CODE HERE!"""
        for parent in parents:
            offspring = parent.copy()
            offspring.mutate() # mutate offspring
            self.gene_pool.append(offspring)
            self.gene_pool.remove(parent)

    def evaluate_population(self):
        """ Evaluate the fitness, and sort the population accordingly."""
        """ YOUR CODE HERE!"""
        avg_fitness = 0
        tmp = []
        for agent in self.gene_pool:
            self.args['pacman'] = agent
            out = runGames(**self.args)
            fitness_score = [o.state.getScore() for o in out]
            tmp.append((fitness_score, agent))

        # Sort populations in increasing fitness-level order
        def sortFitness(val):
            return val[0]
        tmp.sort(key = sortFitness)

        # Add sorted agents to gene pool
        self.gene_pool = []
        for agentPair in tmp:
            (f, agent) = agentPair
            avg_fitness += np.average(f)
            agent.fitness = avg_fitness
            self.gene_pool.append(agent)

        # Calculate average fitness
        avg_fitness /= len(self.gene_pool)
        return avg_fitness

    def select_parents(self, num_parents):
        """ YOUR CODE HERE!"""
        return self.gene_pool[0:num_parents] # assuming they are ordered by their fitness level

    def run(self, num_generations=10):
        display_args = copy.deepcopy(self.args)
        display_args['display'] = self.display_graphics
        display_args['numGames'] = 1

        for i in range(num_generations):
            fitness = self.evaluate_population()
            parents = self.select_parents(self.num_parents)
            self.gene_pool = parents
            self.produce_next_generation(parents)


            # TODO: Print some summaries
            if i % 10 == 0 and i>0:
                print("############################################################")
                print("############################################################")
                print("############################################################")
                print('i', i, fitness)
                display_args['pacman'] = self.gene_pool[0]
                print('best genome!')
                self.gene_pool[0].print_genome()
                runGames(**display_args)
                print("############################################################")
                print("############################################################")
                print("############################################################")

        print('best genome!')
        self.gene_pool[0].print_genome()
        runGames(**display_args)

if __name__ == '__main__':
    args = readCommand( sys.argv[1:] ) # Get game components based on input

    pop_size = 16
    num_parents = int(pop_size/4)+1
    numGames = 3
    num_generations = 10000

    GA = EvolvePacManBT(args, pop_size, num_parents, numGames)
    GA.run(num_generations)


