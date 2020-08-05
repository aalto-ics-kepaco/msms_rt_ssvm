####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####
import os
import argparse
import tensorflow as tf
import numpy as np

from itertools import product
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import GroupKFold

from ssvm.ssvm import StructuredSVMMetIdent
from ssvm.examples.utils import read_data
from ssvm.data_structures import CandidateSetMetIdent
from ssvm.development.utils import get_git_revision_short_hash


HP_C = hp.HParam("C", hp.Discrete([100, 500, 1000, 4000, 16000]))
HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([8]))
HP_NUM_INIT_ACT_VAR = hp.HParam("num_init_act_var", hp.Discrete([6]))
HP_GRID = list(product(HP_C.domain.values, HP_BATCH_SIZE.domain.values, HP_NUM_INIT_ACT_VAR.domain.values))
N_TOTAL_PAR_TUP = len(HP_GRID)


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--param_tuple_index", type=int, choices=list(range(N_TOTAL_PAR_TUP)),
                            help="Index of the parameter tuple for which the evaluation should be run.")
    arg_parser.add_argument("--input_data_dir", type=str,
                            help="Directory containing the ISBM2016 metabolite identification reference data.",
                            default="/home/bach/Documents/doctoral/data/metindent_ismb2016")
    arg_parser.add_argument("--output_dir", type=str,
                            help="Base directory to store the Tensorboard logging files, train and test splits, ...",
                            default="./logs")
    arg_parser.add_argument("--output_dir_reproducible", type=str,
                            help="Directory to store the random-subset and training and test set indices used for the"
                                 "hyper parameter evaluation.",
                            default="./reproducible")
    arg_parser.add_argument("--n_samples", type=float, default=1000,
                            help="Number of training examples to use for the evaluation")
    arg_parser.add_argument("--n_epochs", type=int, default=20)
    arg_parser.add_argument("--max_n_train_candidates", type=int, default=100)
    arg_parser.add_argument("--stepsize", type=str, default="linesearch")
    arg_parser.add_argument("--conv_criteria", type=str, default="rel_duality_gap_decay")

    return arg_parser


def train_test_model(hparams, args, train_summary_writer):
    ssvm = StructuredSVMMetIdent(C=hparams[HP_C], rs=928, max_n_epochs=args.n_epochs,
                                 batch_size=hparams[HP_BATCH_SIZE], conv_criteria=args.conv_criteria,
                                 stepsize=args.stepsize) \
        .fit(X_train, mols_train, candidates=cand, num_init_active_vars_per_seq=hparams[HP_NUM_INIT_ACT_VAR],
             debug_args={"track_objectives": True,
                         "track_topk_acc": True,
                         "track_stepsize": True,
                         "track_dual_variables": True,
                         "train_summary_writer": train_summary_writer})

    acc_test_pred = ssvm.score(X_test, mols_test, candidates=cand, score_type="predicted")
    acc_test_rand = ssvm.score(X_test, mols_test, candidates=cand, score_type="random")
    acc_test_first = ssvm.score(X_test, mols_test, candidates=cand, score_type="first_candidate")

    acc_train_pred = ssvm.score(X_train, mols_train, candidates=cand, score_type="predicted")
    acc_train_rand = ssvm.score(X_train, mols_train, candidates=cand, score_type="random")
    acc_train_first = ssvm.score(X_train, mols_train, candidates=cand, score_type="first_candidate")

    return acc_test_pred, acc_test_rand, acc_test_first, acc_train_pred, acc_train_rand, acc_train_first


if __name__ == "__main__":
    # Read the command line arguments
    args = get_argument_parser().parse_args()

    # Load the data
    X, fps, mols, mols2cand = read_data(args.input_data_dir)

    # Extract a subset of the full training data
    subset = np.sort(
        np.random.RandomState(1989).choice(len(X), size=np.minimum(len(X), args.n_samples).astype(int), replace=False)
    )
    X = X[np.ix_(subset, subset)]
    fps = fps[subset]
    mols = mols[subset]
    print("Number of examples:", len(X))

    # Separate a training and test set: 75% / 25%. If a molecular structure appears multiple times, we add them to the
    # same fold.
    train, test = next(GroupKFold(n_splits=4).split(X, groups=mols))
    X_train = X[np.ix_(train, train)]
    X_test = X[np.ix_(test, train)]
    mols_train = mols[train]
    mols_test = mols[test]
    assert not np.any(np.isin(mols_test, mols_train))

    # Write out indices to reproduce results on the same data subset using IOKR
    np.savetxt(os.path.join(args.output_dir_reproducible, "n_samples=%d" % args.n_samples, "subset.txt"), subset)
    np.savetxt(os.path.join(args.output_dir_reproducible, "n_samples=%d" % args.n_samples, "train.txt"), train)
    np.savetxt(os.path.join(args.output_dir_reproducible, "n_samples=%d" % args.n_samples, "test.txt"), test)

    # Wrap the candidate sets for easier access
    cand = CandidateSetMetIdent(mols, fps, mols2cand, idir=os.path.join(args.input_data_dir, "candidates"),
                                preload_data=True, max_n_train_candidates=args.max_n_train_candidates)

    # Tensorflow training summary log-file
    git_hash = get_git_revision_short_hash()
    tb_log_dir_base = os.path.join(args.output_dir, git_hash)

    # Configure parameter grid
    with tf.summary.create_file_writer(tb_log_dir_base).as_default():
        assert hp.hparams_config(
            hparams=[HP_C, HP_BATCH_SIZE, HP_NUM_INIT_ACT_VAR],
            metrics=[hp.Metric("top-%02d_acc_test_pred" % k, display_name="Top-%02d Accuracy (test, pred, %%)" % k)
                     for k in [1, 5, 10, 20]] +
                    [hp.Metric("top-%02d_acc_test_rand" % k, display_name="Top-%02d Accuracy (test, rand, %%)" % k)
                     for k in [1, 5, 10, 20]] +
                    [hp.Metric("top-%02d_acc_test_first" % k, display_name="Top-%02d Accuracy (test, first, %%)" % k)
                     for k in [1, 5, 10, 20]] +
                    [hp.Metric("top-%02d_acc_train_pred" % k, display_name="Top-%02d Accuracy (train, pred, %%)" % k)
                     for k in [1, 5, 10, 20]] +
                    [hp.Metric("top-%02d_acc_train_rand" % k, display_name="Top-%02d Accuracy (train, rand, %%)" % k)
                     for k in [1, 5, 10, 20]] +
                    [hp.Metric("top-%02d_acc_train_first" % k, display_name="Top-%02d Accuracy (train, first, %%)" % k)
                     for k in [1, 5, 10, 20]]
        ), "Could not create the 'hparam_tuning' Tensorboard configuration."

    # Create a summary write to pass into
    hparams = {
        HP_C: HP_GRID[args.param_tuple_index][0],
        HP_BATCH_SIZE: HP_GRID[args.param_tuple_index][1],
        HP_NUM_INIT_ACT_VAR: HP_GRID[args.param_tuple_index][2]
    }
    sum_wrt = tf.summary.create_file_writer(os.path.join(tb_log_dir_base, "C=%d_bsize=%d_ninit=%d" %
                                                         (HP_GRID[args.param_tuple_index][0],
                                                          HP_GRID[args.param_tuple_index][1],
                                                          HP_GRID[args.param_tuple_index][2])))
    acc = train_test_model(hparams, args, train_summary_writer=sum_wrt)

    with sum_wrt.as_default():
        hp.hparams(hparams)
        for k in [1, 5, 10, 20]:
            tf.summary.scalar("top-%02d_acc_test_pred" % k, acc[0][k - 1], step=1)  # test, pred
            tf.summary.scalar("top-%02d_acc_test_rand" % k, acc[1][k - 1], step=1)  # test, rand
            tf.summary.scalar("top-%02d_acc_test_first" % k, acc[2][k - 1], step=1)  # test, first
            tf.summary.scalar("top-%02d_acc_train_pred" % k, acc[3][k - 1], step=1)  # train, pred
            tf.summary.scalar("top-%02d_acc_train_rand" % k, acc[4][k - 1], step=1)  # train, rand
            tf.summary.scalar("top-%02d_acc_train_first" % k, acc[5][k - 1], step=1)  # train, first
