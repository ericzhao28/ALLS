"""Run battery of experiments on MNIST."""

import queue
import time
import os
from joblib import Parallel, delayed
import sys


def sub_process(exps, SEED_COUNT):
    """Build grid"""
    for e in exps:
        for correction in e[1]:
            for strategy in e[2]:
                for params in e[0]:
                    for seed in range(SEED_COUNT):
                        yield params + "--seed {} --sampling_strategy {} --shift_correction {} ".format(seed, strategy, correction)


def main(experiments, N_PARALLEL=2, SEED_COUNT=5):
    if len(sys.argv) > 1:
        GPU_available = [int(sys.argv[1])]
    else:
        GPU_available = [0]

    experiments = list(sub_process(experiments, SEED_COUNT=SEED_COUNT))
    print("Running %s experiments." % len(experiments))

    GPU_available = GPU_available or []
    N_GPU = len(GPU_available)
    q = queue.Queue(maxsize=N_GPU * N_PARALLEL)
    for _ in range(N_PARALLEL):
        for i in GPU_available:
            q.put(i)


    def runner(cmd):
        """Handle GPU command in queue."""
        gpu = q.get()
        os.system(cmd + " --device %d " % gpu)
        q.put(gpu)


    for e in experiments:
        print(e)
    time.sleep(5)

    Parallel(n_jobs=N_GPU * N_PARALLEL,
             backend="multiprocessing")(delayed(runner)(e) for e in experiments)
