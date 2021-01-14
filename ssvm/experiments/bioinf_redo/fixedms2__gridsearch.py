import sqlite3
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools as it
import argparse

from tensorboard.plugins.hparams import api as hp
from matchms.Spectrum import Spectrum
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, CandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

from msmsrt_scorer.experiments.EA_Massbank.plot_and_table_utils import IDIR as IDIR_EA
from msmsrt_scorer.experiments.CASMI_2016.plot_and_table_utils import IDIR as IDIR_CASMI

#
# HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([4, 8, 16]))
# HP_NUM_INIT_ACT_VAR = hp.HParam("num_init_act_var", hp.Discrete([6]))
# HP_RT_WEIGHT = hp.HParam("rt_weight", hp.Discrete([0.0, 0.5, 1.0]))
# HP_GRID = list(it.product(HP_C.domain.values,
#                           HP_BATCH_SIZE.domain.values,
#                           HP_NUM_INIT_ACT_VAR.domain.values,
#                           HP_RT_WEIGHT.domain.values))
# N_TOTAL_PAR_TUP = len(HP_GRID)


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("eval_set_id", type=int, choices=range(250))
    arg_parser.add_argument("--n_jobs", type=int, default=1)
    arg_parser.add_argument("--debug", action="store_true")
    arg_parser.add_argument("--db_fn", type=str,
                            help="Path to the CASMI database.",
                            default="/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db")
    arg_parser.add_argument("--bioinf_dir", type=str,
                            help="Base directory of the MS + RT framework (published in Bioinformatics) git repository.",
                            default="/home/bach/Documents/doctoral/projects/rt_msms_score_integration_PUBLICATION")
    arg_parser.add_argument("--output_dir", type=str,
                            help="Base directory to store the Tensorboard logging files, train and test splits, ...",
                            default="./logs/fixedms2")
    arg_parser.add_argument("--n_samples_train", type=int, default=1000,
                            help="Number of training sample sequences.")
    arg_parser.add_argument("--n_epochs", type=int, default=5)
    arg_parser.add_argument("--batch_size", type=int, default=8)
    arg_parser.add_argument("--n_init_per_example", type=int, default=6)
    arg_parser.add_argument("--max_n_train_candidates", type=int, default=100)
    arg_parser.add_argument("--stepsize", type=str, default="linesearch")
    arg_parser.add_argument("--ms2scorer", type=str, default="MetFrag_2.4.5__8afe4a14")
    arg_parser.add_argument("--mol_kernel", type=str, default="minmax_numba", choices=["minmax_numba", "minmax"])
    arg_parser.add_argument("--molecule_identifier", type=str, default="inchikey1", choices=["inchikey1", "inchi"])

    return arg_parser


if __name__ == "__main__":
    # ===================
    # Parse arguments
    # ===================
    args = get_argument_parser().parse_args()

    if args.debug:
        HP_C = hp.HParam("C", hp.Discrete([1, 64]))
        HP_RT_WEIGHT = hp.HParam("rt_weight", hp.Discrete([0.4, 0.5]))
    else:
        HP_C = hp.HParam("C", hp.Discrete([1, 2, 4, 8, 16, 32, 64, 128, 256]))
        HP_RT_WEIGHT = hp.HParam("rt_weight", hp.Discrete([0.3, 0.4, 0.5, 0.6, 0.7]))

    # ==================================
    # Get the evaluation dataset from ID
    # ==================================
    if args.eval_set_id in range(0, 50):
        eval_ds = "CASMI_neg"
        eval_idx = args.eval_set_id
    elif args.eval_set_id in range(50, 100):
        eval_ds = "CASMI_pos"
        eval_idx = args.eval_set_id - 50
    elif args.eval_set_id in range(100, 150):
        eval_ds = "EA_neg"
        eval_idx = args.eval_set_id - 100
    else:  # in range(150, 250)
        eval_ds = "EA_pos"
        eval_idx = args.eval_set_id - 150

    # ===================
    # Get list of Spectra
    # ===================
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)

    # Read in spectra and labels
    res = pd.read_sql_query("SELECT spectrum, %s as molecule, rt, challenge FROM challenges_spectra "
                            "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum"
                            "   INNER JOIN molecules m on m.inchi = s.molecule" % args.molecule_identifier, con=db)
    spectra = [Spectrum(np.array([]), np.array([]),
                        {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg, "molecule_id": mol})
               for (spec_id, rt, chlg, mol) in zip(res["spectrum"], res["rt"], res["challenge"], res["molecule"])]
    labels = res["molecule"].to_list()

    db.close()

    # ======================
    # Get training sequences
    # ======================
    # TODO: Split of test for particular eval set id by spectra used --> rest goes to training.

    train, test = next(GroupShuffleSplit(test_size=0.2, random_state=10).split(np.arange(len(spectra)), groups=labels))

    training_sequences = SequenceSample(
        [spectra[idx] for idx in train], [labels[idx] for idx in train],
        RandomSubsetCandidateSQLiteDB(db_fn=args.db_fn, molecule_identifier="inchikey1", random_state=2,
                                      number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
                                      init_with_open_db_conn=False),
        N=args.n_samples_train,
        L_min=5,
        L_max=20,
        random_state=19,
        ms2scorer=args.ms2scorer)

    # ===================
    # Train the SSVM
    # ===================
    print("TRAINING", flush=True)
    summary_writer = tf.summary.create_file_writer(
        os.path.join(args.output_dir, "test_performance", "C=%d_bsize=%d_ninit=%d_rtw=%.1f" %
                     (HP_GRID[args.param_tuple_index][0],
                      HP_GRID[args.param_tuple_index][1],
                      HP_GRID[args.param_tuple_index][2],
                      HP_GRID[args.param_tuple_index][3])))

    with tf.summary.create_file_writer(args.output_dir).as_default():
        assert hp.hparams_config(
            hparams=[HP_C, HP_BATCH_SIZE, HP_NUM_INIT_ACT_VAR, HP_RT_WEIGHT],
            metrics=[hp.Metric("Top-%02d_acc_test" % k, display_name="Top-%02d Accuracy (test, %%)" % k)
                     for k in [1, 5, 10, 20]]
                    # [hp.Metric("Top-01_map_acc_test", display_name="Top-01 Accuracy (MAP, test, %%)")]
        ), "Could not create the 'hparam_tuning' Tensorboard configuration."

    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss="iokr_fps__positive", mol_feat_retention_order="substructure_count",
        mol_kernel=args.mol_kernel, C=hparams[HP_C], step_size=args.stepsize, batch_size=hparams[HP_BATCH_SIZE],
        n_epochs=args.n_epochs, label_loss="tanimoto_loss", random_state=1993,
        retention_order_weight=hparams[HP_RT_WEIGHT], n_jobs=args.n_jobs
    ).fit(training_sequences, n_init_per_example=hparams[HP_NUM_INIT_ACT_VAR], summary_writer=summary_writer)

    # ====================
    # Evaluate performance
    # ====================
    print("EVALUATION", flush=True)
    test_sequences = SequenceSample(
        [spectra[idx] for idx in test], [labels[idx] for idx in test],
        CandidateSQLiteDB(db_fn=args.db_fn, molecule_identifier="inchikey1", init_with_open_db_conn=False),
        N=args.n_samples_test,
        L_min=30,
        L_max=50,
        random_state=19,
        ms2scorer=args.ms2scorer)

    test_acc_tkmm = ssvm.score(test_sequences, stype="topk_mm")
    # test_acc_t1map = ssvm.score(test_sequences, stype="top1_map")

    with summary_writer.as_default():
        hp.hparams(hparams)
        for k in [1, 5, 10, 20]:
            tf.summary.scalar("Top-%02d_acc_test" % k, test_acc_tkmm[k - 1], step=1)  # test, pred
        # tf.summary.scalar("Top-01_map_acc_test", test_acc_t1map, step=1)
