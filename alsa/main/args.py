"""Argument parsing utilites"""

import argparse
import torch

from alsa.nets.mnist import MNISTNet, SimpleMNISTNet
from alsa.nets.cifar import CIFARNet
from alsa.nets.bird import BIRDNet


def get_args(argstring):
    """Argparse specs"""

    use_cuda = torch.cuda.is_available()
    assert use_cuda
    device = 0 if use_cuda else "cpu"
    print("Default device", device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--nabirdstype', type=str, default="grand")
    parser.add_argument('--dataset_cap', type=int, default=3000)
    parser.add_argument('--warmstart_ratio', type=float, default=1.0)
    parser.add_argument('--initial_prop', type=float, default=0.0)
    parser.add_argument('--sample_prop', type=float, default=0.2)
    parser.add_argument('--simple_model',
                        dest='simple_model',
                        action='store_true')
    parser.set_defaults(simple_model=False)

    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--finetune_lr', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--infer_batch_size', type=int, default=512)
    parser.add_argument('--initial_epochs', type=int, default=60)
    parser.add_argument('--partial_epochs', type=int, default=10)
    parser.add_argument('--warm_epochs', type=int, default=1)

    parser.add_argument('--shift_correction', type=str, default="none")
    parser.add_argument('--rlls_reg', type=float, default=1e-3)
    parser.add_argument('--rlls_lambda', type=float, default=1.0)
    parser.add_argument('--no_train_iw', dest='train_iw', action='store_false')
    parser.set_defaults(train_iw=True)
    parser.add_argument('--iterative_iw',
                        dest='iterative_iw',
                        action='store_true')
    parser.set_defaults(iterative_iw=False)

    parser.add_argument('--shift_strategy',
                        type=str,
                        default="dirichlet_target")
    parser.add_argument('--tweak_label', type=int, default=5)
    parser.add_argument('--tweak_prop', type=float, default=0.7)
    parser.add_argument('--dirichlet_alpha', type=float, default=0.6)

    parser.add_argument('--num_batches', type=int, default=20)
    parser.add_argument('--sampling_strategy', type=str, default="maxent")
    parser.add_argument('--diversify', type=str, default="none")
    parser.add_argument('--iwal_normalizer', type=float, default=1e-3)
    parser.add_argument('--vs_size', type=int, default=8)
    parser.add_argument('--bald_size', type=int, default=20)
    parser.add_argument('--no_infer', dest='rlls_infer', action='store_false')
    parser.set_defaults(rlls_infer=True)
    parser.add_argument('--only_rlls_infer',
                        dest='only_rlls_infer',
                        action='store_true')
    parser.set_defaults(only_rlls_infer=False)
    parser.add_argument('--reweight', dest='reweight', action='store_true')
    parser.set_defaults(reweight=False)
    parser.add_argument('--domainsep', dest='domainsep', action='store_true')
    parser.set_defaults(domainsep=False)

    parser.add_argument('--device', type=int, default=device)
    parser.add_argument('--version', type=int, default=17)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--seed_count', type=int, default=1)
    parser.add_argument('--no_log', dest='log', action='store_false')
    parser.set_defaults(log=True)

    if argstring:
        args = parser.parse_args(argstring.split())
    else:
        args = parser.parse_args()

    args.tweak_prop = args.dirichlet_alpha  # temp TODO remove

    return args


def get_net_cls(args):
    """Get network class"""
    if args.dataset == "mnist":
        return SimpleMNISTNet if args.simple_model else MNISTNet
    assert not args.simple_model
    if args.dataset == "cifar":
        return CIFARNet
    if args.dataset == "cifar100":
        return CIFARNet
    if args.dataset == "nabirds":
        if args.simple_model:
            raise ValueError("No simple model for nabirds")
        return BIRDNet
    raise ValueError()
