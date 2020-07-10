import numpy as np
from operator import attrgetter
from copy import copy
from random import shuffle, seed
from model.SupplyChain_thesis import returnTSCC, runSC

seed(123)
np.random.seed(123)

class GenAlg:
    def __init__(self, args, demand, **kwargs):
        self.args = args
        self.n = 20  # number of chromosomes (= pop_size); fixed
        self.l = 4  # length of chromosomes (=agents in supply chain)
        self.par_pop = [Chrom(no=i, args=args) for i in range(1, self.n + 1)]  # parent population
        self.int_pop = []  # intermediate population
        self.pool = []  # mating pool; changes every iteration
        self.cr = kwargs.get('cr', 0.7)  # crossover rate (probability)
        self.mr = kwargs.get('mp', 0.1)  # mutation rate (probability)
        self.x = kwargs.get('mx', 0.2)  # strength of mutation: [(1-x)*s; (1+x)*s]
        self.no_gen = 0
        self.tscc = []  # save tscc of each generation for analysis
        self.rlt = args['rlt']
        self.hcs = args['hcs']
        self.scs = args['scs']
        self.ilt = args['ilt']
        self.RMSilt = args['RMSilt']
        self.demand = demand
        self.rechenberg = kwargs.get("rechenberg", False)
        self.rx = kwargs.get("rx", 0.1)
        self.history = []
        self.success_rate = 0
        self.incs = 0

        for chrom in self.par_pop:
            chrom.evaluate(self.demand)

    def runAlgorithm(self, maxGen):
        """Comment out unwanted configurations!"""
        while self.no_gen < maxGen:
            self.no_gen += 1
            # self.random_crossover()
            self.roulette_crossover()
            if self.rechenberg:
                self.rechenberg_mutation()
            else:
                self.mutation()
            self.evaluation()
            # self.wheel_selection()
            self.elite_selection()
            self.tscc.append(self.par_pop[0].tscc)  # save tscc of current iteration

    def roulette_crossover(self):
        fks = [1 / (1 + chrom.tscc) for chrom in self.par_pop]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        self.pool = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(self.pool) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in self.pool:
                self.pool.append(cn)

        self.pool = [self.par_pop[i] for i in self.pool]

        u = np.random.uniform(0, 1)
        self.int_pop = []
        for i in range(int(np.ceil(self.n / 2))):
            if u <= self.cr:
                cut = np.random.randint(1, self.l)
                cross1 = np.append(self.pool[i * 2].chromosome[:cut], self.pool[i * 2 + 1].chromosome[cut:])
                cross2 = np.append(self.pool[i * 2 + 1].chromosome[:cut], self.pool[i * 2].chromosome[cut:])
                self.int_pop += [Chrom(genes=cross1, args=self.args, l=self.l),
                                 Chrom(genes=cross2, args=self.args, l=self.l)]
            else:
                self.int_pop += self.pool[i * 2:(i * 2 + 1)]

    def random_crossover(self):
        self.pool = copy(self.par_pop)
        shuffle(self.pool)
        self.int_pop = []
        for i in range(int(np.ceil(self.n / 2))):
            u = np.random.uniform(0, 1)
            if u <= self.cr:
                cut = np.random.randint(1, self.l)
                cross1 = np.append(self.pool[i * 2].chromosome[:cut], self.pool[i * 2 + 1].chromosome[cut:])
                cross2 = np.append(self.pool[i * 2 + 1].chromosome[:cut], self.pool[i * 2].chromosome[cut:])
                self.int_pop += [Chrom(genes=cross1, args=self.args), Chrom(genes=cross2, args=self.args)]
            else:
                self.int_pop += self.pool[i * 2:(i * 2 + 1)]

    def mutation(self):
        for c, chrom in enumerate(self.int_pop):
            newGenes = []
            for g, gene in enumerate(chrom.chromosome):
                u = np.random.uniform(0, 1)
                if u <= self.mr:
                    newGenes.append(int(np.floor(gene * (1 - self.x) + gene * 2 * self.x * u)))
                else:
                    newGenes.append(gene)
            self.int_pop[c] = Chrom(genes=np.array(newGenes), args=self.args)

    def rechenberg_mutation(self):
        if self.no_gen != 1:
            if self.history[self.no_gen - 1] > self.history[self.no_gen - 2]:
                self.incs += 1
            self.success_rate = self.incs / (self.no_gen - 1)

        if self.no_gen > 10:
            if self.success_rate < 0.2:
                self.x = self.x ** (1 + self.rx)
            else:
                self.x = self.x ** (1 - self.rx)

        for c, chrom in enumerate(self.int_pop):
            newGenes = []
            for g, gene in enumerate(chrom.chromosome):
                u = np.random.uniform(0, 1)
                if u <= self.mr:
                    newGenes.append(int(np.floor(gene * (1 - self.x) + gene * 2 * self.x * u)))
                else:
                    newGenes.append(gene)
            self.int_pop[c] = Chrom(genes=np.array(newGenes), args=self.args)

    def evaluation(self):
        for chrom in self.int_pop:
            chrom.evaluate(self.demand)

    def wheel_selection(self):
        """Roulette wheel selection"""
        pool = self.int_pop + self.par_pop
        fks = [1 / (1 + chrom.tscc) for chrom in pool]
        sumfk = sum(fks)
        probabilities = np.array([fk / sumfk for fk in fks]).cumsum()

        self.pool = [next(chrom for chrom, val in enumerate(probabilities) if val >= np.random.uniform(0, 1))]
        while len(self.pool) < self.n:
            r = np.random.uniform(0, 1)
            cn = next(chrom for chrom, val in enumerate(probabilities) if val >= r)
            if cn not in self.pool:
                self.pool.append(cn)

        self.par_pop = [pool[i] for i in self.pool]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i

        # Save current best tscc to history
        s = sorted(self.par_pop, key=attrgetter('tscc'))[0]
        self.history.append({"TSCC": s.tscc, "CHROM": s.chromosome})

    def elite_selection(self):
        """Elitist selection"""
        self.par_pop = sorted(self.int_pop + self.par_pop, key=attrgetter('tscc'))[:self.n]

        for i, chrom in enumerate(self.par_pop):
            chrom.no = self.no_gen + i

        self.history.append({"TSCC": self.par_pop[0].tscc, "CHROM": self.par_pop[0].chromosome})


class Chrom:

    def __init__(self, **kwargs):
        self.args = kwargs.get('args')
        self.no = kwargs.get('no', -999)
        self.minRLT = np.array(self.args['rlt']) + np.array(self.args['ilt'][1:].tolist() + [self.args['RMSilt']])
        self.maxRLT = np.cumsum(self.minRLT[::-1])[::-1]
        self.lowerU = 20
        self.upperU = 60
        self.chromosome = kwargs.get('genes', self.generateChromosome())
        self.tscc = 0
        self.hcs = self.args['hcs']
        self.scs = self.args['scs']

    def generateChromosome(self):
        """generate initial chromosomes"""
        lower = self.lowerU * self.minRLT
        upper = self.upperU * self.maxRLT
        return np.array([np.random.randint(l, u + 1) for l, u in zip(lower, upper)])

    def evaluate(self, demand):
        """Run the SC model and evaluate TSCC"""
        # self.tscc = returnTSCC(self.chromosome)   # used for testing
        self.tscc = runSC(self.chromosome, args=self.args, demand=demand)
