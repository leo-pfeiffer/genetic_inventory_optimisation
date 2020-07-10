import numpy as np
from operator import attrgetter
from tqdm import tqdm
from random import uniform, shuffle
from copy import copy
from model.GenericGA.FitnessFunctions import multimodal, rosenbrock


class SimpleGenAlg:
    def __init__(self, args, m):
        self.args = args
        self.n = 4  # number of chromosomes (= pop_size)
        self.l = 2  # length of chromosomes (=agents in supply chain)
        self.par_pop = [Chrom(no=i, args=args, l=self.l) for i in range(1, self.n + 1)]  # parent population
        self.int_pop = []  # intermediate population
        self.pool = []  # mating pool; changes every iteration
        self.cr = 0.7  # crossover rate (probability)
        self.mr = m  # mutation rate (probability) default 0.7
        self.x = 0.2  # parameter for mutation
        self.no_gen = 0
        self.fitness = []  # save fitness of each generation for analysis
        self.search = []    # hold history of best solution

        for chrom in self.par_pop:
            chrom.evaluate()

    def runAlgorithm(self, maxGen):
        if __name__ == '__main__':
            pbar = tqdm(maxGen)
        while self.no_gen < maxGen:
            self.no_gen += 1
            self.crossover()
            self.mutation()
            self.survival() # elitist
            # self.survival2()  # roulette wheel

            self.fitness.append(self.par_pop[0].fitness)  # save fitness of current iteration
            self.search.append(self.par_pop[0].chromosome)  # save best chromosome

            if __name__ == '__main__':
                pbar.update(1)

    def crossover(self):
        self.pool = copy(self.par_pop)
        shuffle(self.pool)
        for i in range(int(np.ceil(self.n/2))):
            cut = np.random.randint(1, self.l)
            cross1 = np.append(self.pool[i*2].chromosome[:cut], self.pool[i*2+1].chromosome[cut:])
            cross2 = np.append(self.pool[i*2+1].chromosome[:cut], self.pool[i*2].chromosome[cut:])
            self.int_pop += [Chrom(genes=cross1, args=self.args, l=self.l), Chrom(genes=cross2, args=self.args, l=self.l)]

    def mutation(self):
        for c, chrom in enumerate(self.int_pop):
            newGenes = []
            for g, gene in enumerate(chrom.chromosome):
                u = np.random.uniform(0, 1)
                if u <= self.mr:
                    newGenes.append(gene * (1 - self.x) + gene * 2 * self.x * u)
                else:
                    newGenes.append(gene)
            self.int_pop[c] = Chrom(genes=np.array(newGenes), args=self.args, l=self.l)
            self.int_pop[c].evaluate()

    def survival2(self):
        """Roulette wheel selection"""
        pool = self.int_pop + self.par_pop
        fks = [chrom.fitness for chrom in pool]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        new_par = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(new_par) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in new_par:
                new_par.append(cn)

        self.par_pop = [pool[i] for i in new_par]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i

    def survival(self):
        """elitist selection"""
        pool = self.int_pop + self.par_pop

        self.par_pop = sorted(pool, key=attrgetter('fitness'))[:self.n]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i


class Chrom:

    def __init__(self, **kwargs):
        self.args = kwargs.get('args')
        self.no = kwargs.get('no', -999)
        self.l = kwargs.get('l', 2)
        self.chromosome = kwargs.get('genes', self.generateChromosome())
        self.fitness = np.inf

    def generateChromosome(self):
        """generate initial chromosomes"""
        return np.array([uniform(self.args['lower'], self.args['upper']) for i in range(self.l)])

    def evaluate(self):
        """Call fitness function"""
        self.fitness = multimodal(self.chromosome, a=1, b=100)
