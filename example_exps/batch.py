"""Run battery of experiments on MNIST."""

from main import *
from settings import *

SEED_COUNT = 10
N_PARALLEL = 2

def sub_process(exps, SEED_COUNT):
    """Build grid"""
    for seed in range(SEED_COUNT):
        for e in exps:
            for correction in e[1]:
                for strategy in e[2]:
                    for params in e[0]:
                        yield params + "--seed {} --sampling_strategy {} --shift_correction {} ".format(seed, strategy, correction)

experiments = [
    # How is RLLS being used?
    [(cifar100_common + warmstart,),
     ("none",
      "rlls --iterative_iw --no_train_iw --reweight --no_infer",
      "rlls --iterative_iw --no_train_iw --reweight --only_rlls_infer",
      "rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald",
     )
    ],
    # Which RLLS heuristics are better?
    [(cifar100_common + warmstarttar,),
     ("none",
      "rlls",
      "rlls --no_train_iw",
      "rlls --reweight --iterative_iw",
      "rlls --iterative_iw",
      "rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald",
     )
    ],
    # Online experiments
    [(cifar100_common + online, cifar_common + online,),
     ("none",),
     (
      "bald",
      "random",
     )
    ],
    [(cifar100_common + online, cifar_common + online,),
     ("rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald --diversify guess",
     )
    ],
    # Warmstart experiments ablation
    [(cifar100_common + warmstartsrc, cifar100_common + warmstarttar,),
     ("none",),
     (
      "bald",
      "random",
     )
    ],
    [(cifar100_common + warmstartsrc, cifar100_common + warmstarttar,),
     ("rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald --diversify guess",
     )
    ],
    # Warmstart experiments
    [(cifar100_common + warmstart,),
     ("none",),
     (
      "bald",
      "maxent",
      "margin",
      "random",
     )
    ],
    [(cifar100_common + warmstart,),
     ("rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald --diversify guess",
      "maxent --diversify guess",
      "margin --diversify guess",
     )
    ],
    # Warmstart alphas
    [(cifar100_common + warmstart1, cifar100_common + warmstart2, cifar100_common + warmstart3, cifar100_common + warmstart,),
     ("none",
      "rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald --diversify guess",
      "bald",
      "random",
     )
    ],
    # Online experiments
    [(nabirds_common + naregion,),
     ("none",),
     (
      "bald",
      "random",
     )
    ],
    [(nabirds_common + naregion,),
     ("rlls --iterative_iw --no_train_iw --reweight",),
     (
      "bald --diversify guess",
     )
    ],
]

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
