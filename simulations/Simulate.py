import concurrent.futures
import numpy as np
import pandas as pd
import random
from model.GenAlg_thesis import GenAlg
from model.SCsettings_thesis import s1, s2, s3, s4, s5, s6, demandSample, randomArgsBased
from model.RandomSearch import RandomSearch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from warnings import warn

random.seed(123)
np.random.seed(123)


def simulate(method):
    args = (n_it, demand, arg, mx, mp, cr, max_gen, rad)
    if method == "GA":
        fun = ga_process_wrapper
    elif method == "RS":
        fun = rs_process_wrapper
    else:
        warn("Select RS or GA as method!")
        exit()

    tscc, history = fun(args)

    tscc = pd.DataFrame(tscc).T
    tscc['Mean'] = tscc.apply(np.mean, 1)

    return tscc, history


def simulate_multiproc(method):
    """Does not support history."""
    tscc = []
    chromosomes = []

    if method == "GA":
        fun = ga_process_wrapper
    elif method == "RS":
        fun = rs_process_wrapper
    else:
        warn("Select RS or GA as method!")
        exit()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
        args = ((len(d), d, arg, mx, mp, cr, max_gen, rad) for d in demands)
        return_split = executor.map(fun, args)

        for ret in return_split:
            chromosomes += ret[0]
            tscc += ret[1]

    tscc = pd.DataFrame(tscc).T
    tscc['Mean'] = tscc.apply(np.mean, 1)
    return tscc, chromosomes


def rs_process(its, demand, arg, mx, mp, cr, max_gen, rad):
    random.seed(123)
    np.random.seed(123)
    history = []
    tscc = []
    for i in tqdm([*range(its)]):
        RS = RandomSearch(args=arg, demand=demand[i], rad=rad)
        RS.runAlgorithm(maxGen=max_gen)
        tscc.append(RS.tscc)
        history.append(RS.history)
    return tscc, history


def rs_process_wrapper(args):
    return rs_process(*args)


def ga_process(its, demand, arg, mx, mp, cr, max_gen, rad):
    random.seed(123)
    np.random.seed(123)
    history = []
    tscc = []
    for i in tqdm([*range(its)]):
        GA = GenAlg(args=arg, demand=demand[i], mx=mx, mp=mp, cr=cr, rechenberg=False)
        GA.runAlgorithm(maxGen=max_gen)
        tscc.append(GA.tscc)
        history.append(GA.history)
    return tscc, history


def ga_process_wrapper(args):
    return ga_process(*args)


def plot(tscc, history):
    history = [(x['TSCC'], x['CHROM']) for x in sum(history, [])]
    history_sorted = sorted(history, key=lambda tup: tup[0])
    best = [(x[0], x[1].tolist()) for x in history_sorted][0]
    fig = plt.figure(figsize=(9, 6), dpi=300)
    fig.tight_layout(pad=0.1)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([*range(len(tscc))], tscc.Mean.values.tolist())[0]
    ax.set_ylabel("TSCC")
    ax.set_xlabel("Generation number")

    text = "Best policy:\nRetailer: {}, Distributer: {},\nManufacturer: {}, " \
           "Supplier: {}\n--------------------------------------------------\nTSCC: {}"
    text = text.format(best[1][0], best[1][1], best[1][2], best[1][3], best[0])
    ax.text(len(tscc) - 1, tscc.Mean.values[0], text, fontsize=10, va="top", ha="right")

    plt.savefig(filename+".png")


if __name__ == "__main__":
    n_it = 30
    T = 1200
    lower = 20
    upper = 60
    tasks = 1
    max_gen = 200
    mx = 0.2
    mp = 0.8
    cr = 0.7
    rad = 0.9
    filename = "GA2/GA2_s4"
    arg = s4
    # arg = randomArgsBased(s1, ilt=np.random.randint(1, 32, 4),
    #                      rlt=np.random.randint(1, 32, 4), RMSilt=np.random.randint(1, 3))
    demand = demandSample(T, lower, upper, n_it, antithetic=True)

    t = time.time()
    tscc, history = simulate(method="GA")
    tscc.to_csv(filename+"_tscc.csv", header=True)
    pd.DataFrame(history).T.to_csv(filename+"_history.csv", header=True)
    plot(tscc, history)
    print(time.time() - t)
