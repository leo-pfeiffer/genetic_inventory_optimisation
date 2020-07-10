from model.GenericGA.GenericGenAlg import SimpleGenAlg
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from model.GenericGA.FitnessFunctions import multimodal


mrs = np.arange(0.05, 1, 0.01).tolist()
space = np.arange(5, 11, 1).tolist()
simulations = 25
delta = 0.1
df = pd.DataFrame()
labels = ["No optimum found", r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$']


def binner(x, delta):
    v1 = 3 * math.sqrt(3 / 10)
    v2 = math.sqrt(23 / 10)
    min1 = multimodal([v1, -v1], 0, 0)
    min2 = multimodal([-v1, v1], 0, 0)
    min3 = multimodal([-v2, -v2], 0, 0)
    min4 = multimodal([v2, v2], 0, 0)

    if abs(x - min1)/min1 <= delta:
        return 1
    elif abs(x - min2)/min2 <= delta:
        return 2
    elif abs(x - min3)/min3 <= delta:
        return 3
    elif abs(x - min4)/min4 <= delta:
        return 4
    else:
        return 0


def analyseSimulation(**kwargs):
    df = kwargs.get('df')
    simulations = kwargs.get('simulations')
    data = pd.DataFrame(columns=[0, 1, 2, 3, 4])
    params = []
    #for m in mrs:
    for m in [0.72]:
        #for s in space:
        for s in space:
            params.append((m, s))
            freq = df[(df.MR == m) & (df.Search == s)].Bin.value_counts()/simulations
            data = data.append(freq.to_dict(), ignore_index=True).fillna(0)

    index = data[data[2] > 0.5].index
    for i in index:
        print(params[i])
        print(data.iloc[i])
        print("---------")

    return data


if __name__ == "__main__":
    pbar = tqdm(len(space))
    #for m in mrs:
    for m in [0.72]:
        #for s in space:
        for s in [2.048]:
            for i in range(simulations):
                args = {"lower": -s,
                        "upper": s}
                max_gen = 100
                GA = SimpleGenAlg(args=args, m=m)
                GA.runAlgorithm(maxGen=max_gen)
                fitness = GA.fitness[-1]
                b = binner(fitness, delta)
                x = GA.search[-1][0]
                y = GA.search[-1][1]
                data = {'MR': m, 'Search': s, 'Sim': i, 'Bin': b, 'x': x, 'y': y, 'fitness': fitness}
                df = df.append(data, ignore_index=True)

                pbar.update(1)

    df[["x", "y", "fitness"]].plot()
    plt.show()

    """
    # Plot and save output
    df.to_csv("SimulationResults.csv")
    data = analyseSimulation(df=df, simulations=simulations)
    data.index = space
    data.to_csv("Solutions_MR072.csv", index=True, index_label="MR")
    sss = [r'$[{}, {}]$'.format(-x, x) for x in space if x%5==0]
    plt.figure(figsize=(8, 4), dpi=300)
    plt.stackplot(space, data.loc[:, 0], data.loc[:, 1], data.loc[:, 2], data.loc[:, 3], data.loc[:, 4],
                  labels=labels)
    plt.legend(loc="upper left")
    plt.xticks(ticks=[x for x in space if x%5==0], labels=sss)
    plt.xlabel("Search space")
    plt.ylabel("Relative Frequency of Solution")
    plt.margins(0,0)
    plt.savefig("Solutions_MR072.png")
    plt.show()
    """