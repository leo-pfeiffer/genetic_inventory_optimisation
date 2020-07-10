import numpy as np
import pandas as pd
import random

random.seed(123)
np.random.seed(123)

# Settings bachelor thesis

s1 = {
    'hcs': np.array([4, 3, 2, 1]),
    'scs': np.array([8, 6, 4, 2]),
    'rlt': np.array([2, 3, 4, 5]),
    'ilt': np.array([1, 2, 3, 4]),
    'RMSilt': 5
}

s2 = {
    'hcs': np.array([4, 3, 2, 1]),
    'scs': np.array([8, 6, 4, 2]),
    'rlt': np.array([2, 4, 16, 32]),
    'ilt': np.array([1, 3, 9, 18]),
    'RMSilt': 24
}

s3 = {
    'hcs': np.array([12, 8, 4, 1]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([2, 3, 4, 5]),
    'ilt': np.array([1, 2, 3, 4]),
    'RMSilt': 5
}

s4 = {
    'hcs': np.array([12, 8, 4, 1]),
    'scs': np.array([24, 12, 6, 3]),
    'rlt': np.array([2, 4, 16, 32]),
    'ilt': np.array([1, 3, 9, 18]),
    'RMSilt': 24
}

# test large costs
s5 = {
    'hcs': np.array([240, 160, 80, 20]),
    'scs': np.array([480, 240, 120, 60]),
    'rlt': np.array([2, 3, 4, 5]),
    'ilt': np.array([1, 2, 3, 4]),
    'RMSilt': 5
}

# test large lead times
s6 = {
    'hcs': np.array([4, 3, 2, 1]),
    'scs': np.array([8, 6, 4, 2]),
    'rlt': np.array([200, 100, 150, 2]),
    'ilt': np.array([25, 100, 66, 39]),
    'RMSilt': 33
}


def randomArgsBased(arg, **kwargs):
    """Creates random arguments. Not used in thesis"""
    ilt = kwargs.get('ilt', arg['ilt'])
    rlt = kwargs.get('rlt', arg['rlt'])
    RMSilt = kwargs.get('RMSilt', arg['RMSilt'])
    arg_out = arg
    arg_out['ilt'] = ilt
    arg_out['rlt'] = rlt
    arg_out['RMSilt'] = RMSilt
    print(arg_out)
    return arg_out


def demandSample(T, lower, upper, n_it, antithetic):
    """Creates antithetic demand sampple"""
    if antithetic:
        sample = [np.random.randint(lower, upper + 1, T).tolist() for i in range(int(np.floor(n_it / 2)))]
        anti_sample = [[(lower + upper) - u for u in demand] for demand in sample]
        return sample + anti_sample

    else:
        return [np.random.randint(lower, upper + 1, T).tolist() for i in range(n_it)]


def Output(results, ought, elapsed):
    df = pd.DataFrame(results, columns=["Mean", "SD"]).astype(int)
    ought_df = pd.DataFrame(ought, columns=["Mean", "SD"]).astype(int)
    delta_perc = np.round((df / ought_df - 1), 4)
    delta_abs = (df - ought_df).astype(int)
    print("\nSummary\nElapsed Time:", elapsed, "\n\n", df, "\n", delta_perc, "\n", delta_abs)
    return df, ought_df, delta_perc, delta_abs
