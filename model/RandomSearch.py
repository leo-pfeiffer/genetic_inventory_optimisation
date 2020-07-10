import numpy as np
from random import randint, seed
from model.SupplyChain_thesis import runSC

seed(123)
np.random.seed(123)


class RandomSearch:
    def __init__(self, args, demand, **kwargs):
        self.args = args
        self.l = 4  # length of chromosomes (=agents in supply chain)
        self.tscc = []  # save tscc of each generation for analysis
        self.history = []  # save solution of every generation
        self.no_gen = 0
        self.a = kwargs.get('rad', 0.9)   # relative search radius

        self.rlt = args['rlt']
        self.hcs = args['hcs']
        self.scs = args['scs']
        self.ilt = args['ilt']
        self.RMSilt = args['RMSilt']
        self.demand = demand

        self.minRLT = np.array(self.rlt) + np.array(self.ilt[1:].tolist() + [self.RMSilt])
        self.maxRLT = np.cumsum(self.minRLT[::-1])[::-1]
        self.lowerU = 20
        self.upperU = 60

        self.LB = self.lowerU * self.minRLT
        self.UB = self.upperU * self.maxRLT

        self.hcs = self.args['hcs']
        self.scs = self.args['scs']

        self.parent = self.generateRandom()
        self.par_tscc = runSC(self.parent, args=args, demand=self.demand)
        self.child = []
        self.child_tscc = np.inf

    def runAlgorithm(self, maxGen):
        while self.no_gen < maxGen:
            self.no_gen += 1
            self.alter()
            self.evaluateChild()
            self.selection()
            self.tscc.append(self.par_tscc)  # save tscc of current iteration

    def generateRandom(self):
        """generate initial solution"""
        return np.array([np.random.randint(l, u + 1) for l, u in zip(self.LB, self.UB)])

    def alter(self):
        upperX = [int(min(x*(1+self.a), self.UB[i])) for i, x in enumerate(self.parent)]
        lowerX = [int(max(x*(1-self.a), self.LB[i])) for i, x in enumerate(self.parent)]
        self.child = [randint(lowerX[i], upperX[i]) for i, x in enumerate(self.parent)]

    def evaluateChild(self):
        self.child_tscc = runSC(self.child, args=self.args, demand=self.demand)

    def selection(self):
        if self.child_tscc < self.par_tscc:
            self.parent = self.child
            self.par_tscc = self.child_tscc

        self.history.append({"TSCC": self.par_tscc, "CHROM": np.array(self.parent)})
