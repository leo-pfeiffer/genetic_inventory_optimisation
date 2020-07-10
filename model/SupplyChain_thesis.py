import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

random.seed(123)
np.random.seed(123)


class Agent:

    def __init__(self, no, basestock, ilt, rlt, RMSilt, hcs, scs):
        self.no = no
        self.T = 1200
        self.basestock = basestock
        self.ilt = ilt
        self.RMSilt = RMSilt
        self.rlt = rlt
        self.holdingcost_rate = hcs
        self.shortagecost_rate = scs
        self.holdingcost_is = 0
        self.shortagecost_is = 0
        self.onHandInventory = basestock
        self.onOrderInventory = 0
        self.backlog = 0
        self.order = [0] * int(np.floor(self.T * 1.5))  # order from downstream
        self.receive = [0] * int(np.floor(self.T * 1.5))  # received from upstream; account for schedule beyond T


class SupplyChain:

    def __init__(self, agents, demand):
        self.agents = agents
        self.T = 1200
        self.demand = demand
        self.scc = [0] * self.T
        self.tscc = 0
        self.h_costs = []
        self.s_costs = []
        self.inventory = []

    def simulate(self):
        for t in range(0, self.T):

            # update the order info from
            for i, agent in enumerate(self.agents):

                # define upstream and downstream
                if agent.no != 0:
                    downstream = self.agents[agent.no - 1]
                if agent.no != 3:
                    upstream = self.agents[agent.no + 1]

                # scheduled receipt of material at agent
                agent.onHandInventory += agent.receive[t]
                agent.onOrderInventory -= agent.receive[t]

                # backlog is filled as much as possible
                if agent.backlog > 0:
                    newBacklog = max(agent.backlog - agent.onHandInventory, 0)

                    # for all agents except retailer, schedule reception for downstream agent and do shipment
                    # end customer receives order immediately (no need to be scheduled)
                    if agent.no != 0:
                        downstream.receive[t + downstream.rlt] += agent.backlog - newBacklog

                    # update inventory information
                    agent.onHandInventory = max(agent.onHandInventory - agent.backlog, 0)
                    agent.backlog = newBacklog

                # agents receive demand/order quantity from immediate downstream member
                if agent.no == 0:
                    agent.order[t] = self.demand[t]  # in theory: self.demand[t - agent.ilt] but this is equivalent

                # send order to upstream -> received immediately
                # calculate end of period onHandInventory
                shipment = min(agent.order[t], agent.onHandInventory)

                # order fulfillment. demand/order of t is fulfilled based on inventory
                # if not all can be fulfilled => backlog
                if agent.no != 0:
                    downstream.receive[t + downstream.rlt] += shipment

                agent.onHandInventory -= shipment
                agent.backlog += agent.order[t] - shipment

                orderQuantity = max(agent.basestock + agent.backlog - agent.onHandInventory - agent.onOrderInventory, 0)

                # orders are received in t + ILT
                if agent.no != 3:
                    upstream.order[t + upstream.ilt] = orderQuantity
                else:
                    # supplier receives order from infinite source after rlt + ILT
                    agent.receive[t + agent.rlt + agent.RMSilt] += orderQuantity

                # update inventory and calculate local costs
                agent.onOrderInventory += orderQuantity
                agent.holdingcost_is = agent.holdingcost_rate * agent.onHandInventory
                agent.shortagecost_is = agent.shortagecost_rate * agent.backlog

            # calculate supply chain contexts
            self.scc[t] = sum([x.holdingcost_is + x.shortagecost_is for x in self.agents])

            self.h_costs.append([x.holdingcost_is for x in self.agents])
            self.s_costs.append([x.shortagecost_is for x in self.agents])
            self.inventory.append([x.onHandInventory for x in self.agents])

        self.tscc = np.array(self.scc[0:]).cumsum()[-1]


def returnTSCC(chromosome):
    """Used for testing."""
    goal = [12, 13, 4, 17]
    return sum([abs(x - y) for x, y in zip(chromosome, goal)])


def runSC(chromosome, args, **kwargs):
    rlt = args['rlt']
    hcs = args['hcs']
    scs = args['scs']
    ilt = args['ilt']
    RMSilt = args['RMSilt']
    demand = kwargs.get('demand', np.random.randint(20, 61, 1200))
    agents = []

    for i, chrom in enumerate(chromosome):
        agents.append(Agent(no=i, basestock=chrom, rlt=rlt[i], hcs=hcs[i], RMSilt=RMSilt, scs=scs[i], ilt=ilt[i]))

    S = SupplyChain(agents=agents, demand=demand)
    S.simulate()

    return S.tscc
