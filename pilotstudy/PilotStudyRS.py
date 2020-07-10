import concurrent.futures
import numpy as np
import pandas as pd
from model.RandomSearch import RandomSearch
from model.SCsettings_thesis import s1, s2, s3, s4, demandSample
from tqdm import tqdm
import time
import random
import datetime

n_it = 20
T = 1200
lower = 20
upper = 60
tasks = 1
max_gen = 25
chromosomes = []

demand = demandSample(T, lower, upper, n_it, antithetic=True)

results = pd.DataFrame(columns=["Rad", "TSCC"])

iterations = [*range(n_it)]

rads = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


def rs_process(its, demand, arg, rad, max_gen):
    random.seed(123)
    np.random.seed(123)
    chromosomes = []
    tscc = []
    for i in tqdm([*range(its)]):
        RS = RandomSearch(args=arg, demand=demand[i], rad=rad)
        RS.runAlgorithm(maxGen=max_gen)
        tscc.append(RS.tscc)
        chromosomes.append(np.array(RS.parent))
    return chromosomes, tscc


def rs_process_wrapper(args):
    return rs_process(*args)


t0 = time.time()

s = 0
for arg in [s1, s2, s3, s4]:
    s += 1
    for rad in rads:
        start = time.time()

        with concurrent.futures.ProcessPoolExecutor() as executor:
            demands = [a.tolist() for a in np.array_split(np.array(demand), tasks)]
            args = ((len(d), d, arg, rad, max_gen) for d in demands)
            return_split = executor.map(rs_process_wrapper, args)
            tscc = []
            chromosomes = []
            for ret in return_split:
                chromosomes += ret[0]
                tscc += ret[1]

        avg_tscc = np.mean(tscc, axis=0)

        tscc = avg_tscc[-1]
        elapsed = time.time() - start

        row = {"Rad": rad, "TSCC": tscc}
        print(elapsed, "\n", row, "\n")
        print(datetime.datetime.now().strftime("%H:%M:%S"))
        results = results.append(row, ignore_index=True)

    if s == 1:
        filename = "results/RS/RS_S1.csv"
    elif s == 2:
        filename = "results/RS/RS_S2.csv"
    elif s == 3:
        filename = "results/RS/RS_S3.csv"
    elif s == 4:
        filename = "results/RS/RS_S4.csv"
    results.to_csv(filename, header=True, index=True)
    print("S{} done".format(s))
    print("-----------Next iteration-------------")
    elapsed = (time.time() - t0) / 60
