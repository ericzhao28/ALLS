"""Batch Settings"""

cifar100_common = ("python3 -m alsa.main.replicate --batch_size 128 "
                   "--gamma 0.25 --rlls_reg 2e-06 --initial_epochs 80 "
                   "--partial_epochs 10 --lr 0.1 --finetune_lr 0.02 "
                   "--num_batches 30 --version 301 --dataset cifar100 "
                   "--dataset_cap 40000 ")
cifar_common = ("python3 -m alsa.main.replicate --batch_size 128 "
                "--gamma 0.25 --rlls_reg 2e-06 --initial_epochs 80 "
                "--partial_epochs 10 --lr 0.1 --finetune_lr 0.02 "
                "--num_batches 30 --version 301 --dataset cifar "
                "--dataset_cap 40000 ")
nabirds_common = ("python3 -m alsa.main.replicate --batch_size 128 "
                  "--gamma 0.25 --rlls_reg 2e-06 --initial_epochs 80 "
                  "--partial_epochs 10 --lr 0.1 --finetune_lr 0.02 "
                  "--num_batches 20 --version 301 --dataset nabirds "
                  "--dataset_cap 30000 ")

nachild = ("--sample_prop 0.4 --warmstart_ratio 1.0 --nabirdstype child --shift_strategy trulynone ")
naregion = ("--sample_prop 0.4 --warmstart_ratio 1.0 --nabirdstype grand --shift_strategy trulyregion --dirichlet_alpha 0.1 ")

warmstart = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
            "--shift_strategy dirichlet_mix --dirichlet_alpha 0.7 ")
warmstart1 = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
              "--shift_strategy dirichlet_mix --dirichlet_alpha 5.0 ")
warmstart2 = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
              "--shift_strategy dirichlet_mix --dirichlet_alpha 1.0 ")
warmstart3 = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
              "--shift_strategy dirichlet_mix --dirichlet_alpha 0.1 ")
warmstarttar = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
                "--shift_strategy dirichlet_target "
                "--dirichlet_alpha 0.4 ")
warmstartsrc = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
                "--shift_strategy dirichlet --dirichlet_alpha 1.2 ")

online = ("--sample_prop 0.4 --warmstart_ratio 0.5 "
          "--shift_strategy dirichlet_online --dirichlet_alpha 3.0 ")
