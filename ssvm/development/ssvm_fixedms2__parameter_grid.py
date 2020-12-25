import sqlite3
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools as it
import argparse


from tensorboard.plugins.hparams import api as hp
from matchms.Spectrum import Spectrum

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

HP_C = hp.HParam("C", hp.Discrete([1, 2, 8, 32]))
HP_BATCH_SIZE = hp.HParam("batch_size", hp.Discrete([4, 8, 16]))
HP_NUM_INIT_ACT_VAR = hp.HParam("num_init_act_var", hp.Discrete([3, 6, 9]))
HP_GRID = list(it.product(HP_C.domain.values, HP_BATCH_SIZE.domain.values, HP_NUM_INIT_ACT_VAR.domain.values))
N_TOTAL_PAR_TUP = len(HP_GRID)


def get_argument_parser() -> argparse.ArgumentParser:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--param_tuple_index", type=int, choices=list(range(N_TOTAL_PAR_TUP)),
                            help="Index of the parameter tuple for which the evaluation should be run.")
    arg_parser.add_argument("--db_fn", type=str,
                            help="Path to the CASMI database.",
                            default="/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db")
    arg_parser.add_argument("--output_dir", type=str,
                            help="Base directory to store the Tensorboard logging files, train and test splits, ...",
                            default="./logs")
    arg_parser.add_argument("--n_samples", type=float, default=500,
                            help="Number of training examples to use for the evaluation")
    arg_parser.add_argument("--n_epochs", type=int, default=5)
    arg_parser.add_argument("--max_n_train_candidates", type=int, default=100)
    arg_parser.add_argument("--stepsize", type=str, default="linesearch")

    return arg_parser


if __name__ == "__main__":
    tf_summary_base_dir = "/home/bach/Documents/doctoral/projects/rt_msms_ssvm/src/ssvm/development/logs"
    DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"

    # ===================
    # Get list of Spectra
    # ===================
    db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

    # Read in spectra and labels
    res = pd.read_sql_query("SELECT spectrum, inchikey1 as molecule, rt, challenge FROM challenges_spectra "
                            "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum"
                            "   INNER JOIN molecules m on m.inchi = s.molecule", con=db)
    spectra = [Spectrum(np.array([]), np.array([]),
                        {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg, "molecule_id": mol})
               for (spec_id, rt, chlg, mol) in zip(res["spectrum"], res["rt"], res["challenge"], res["molecule"])]
    labels = res["molecule"].to_list()

    db.close()

    # ===================
    # Setup a SSVM
    # ===================
    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss="iokr_fps__positive", mol_feat_retention_order="substructure_count",
        mol_kernel="minmax_numba", C=2, step_size="linesearch", batch_size=16, n_epochs=1, label_loss="tanimoto_loss",
        random_state=1993, retention_order_weight=1.0)

    N = 250
    seq_sample = SequenceSample(
        spectra, labels,
        RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey1", random_state=2,
                                      number_of_candidates=50, include_correct_candidate=True),
        N=N, L_min=10,
        L_max=20, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

    seq_sample_train, seq_sample_test = seq_sample.get_train_test_split(cv=4)
    print(len(seq_sample_train[0]))

    summary_writer = tf.summary.create_file_writer(os.path.join(tf_summary_base_dir, "test_performance",
                                                                "%d" % np.random.randint(1000)))

    ssvm.fit(seq_sample_train, n_init_per_example=3, summary_writer=None)

    print(ssvm.score(seq_sample_test, stype="top1_map"))

    # TODO: We somehow should ensure that the database connection is always closed.
    seq_sample.candidates.close()

