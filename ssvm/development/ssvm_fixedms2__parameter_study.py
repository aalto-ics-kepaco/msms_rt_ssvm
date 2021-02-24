import os
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List, Tuple
from matchms.Spectrum import Spectrum
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

# ================
# Setup the Logger
LOGGER = logging.getLogger("parameter_study")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def load_spectra_and_labels(dbfn: str, molecule_identifier: str) -> Tuple[List[Spectrum], List[str]]:
    """
    Loads all spectra ids, retention times and ground truth labels.
    """
    db = sqlite3.connect("file:" + dbfn + "?mode=ro", uri=True)

    # Read in spectra and labels
    res = pd.read_sql_query("SELECT spectrum, %s as molecule, rt, challenge FROM challenges_spectra "
                            "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum"
                            "   INNER JOIN molecules m on m.inchi = s.molecule" % molecule_identifier, con=db)
    spectra = [Spectrum(np.array([]), np.array([]),
                        {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg, "molecule_id": mol})
               for (spec_id, rt, chlg, mol) in zip(res["spectrum"], res["rt"], res["challenge"], res["molecule"])]
    labels = res["molecule"].to_list()

    db.close()

    return spectra, labels


def get_cli_arguments() -> argparse.Namespace:
    """
    Set up the command line input argument parser
    """
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--db_fn", type=str, help="Path to the CASMI database.",
        default="/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db")
    arg_parser.add_argument(
        "--output_dir", type=str, help="Base directory to store the Tensorboard logging files etc.",
        default="./logs/parameter_study")

    arg_parser.add_argument("parameter_to_study", type=str)
    arg_parser.add_argument("parameter_grid", type=str, nargs="+")

    arg_parser.add_argument("--n_samples_train", type=int, default=256, help="Number of training sample sequences.")
    arg_parser.add_argument("--n_samples_test", type=int, default=48, help="Number of test / validation sample sequences.")

    # =======================
    # SSVM default parameters
    # =======================
    arg_parser.add_argument("--n_epochs", type=int, default=5)
    arg_parser.add_argument("--batch_size", type=int, default=16)
    arg_parser.add_argument("--n_init_per_example", type=int, default=6)
    arg_parser.add_argument("--step_size_approach", type=str, default="linesearch_parallel")
    arg_parser.add_argument("--mol_kernel", type=str, default="minmax", choices=["minmax_numba", "minmax"])
    arg_parser.add_argument("--C", type=float, default=4)
    arg_parser.add_argument("--label_loss", type=str, default="minmax_loss")
    arg_parser.add_argument("--mol_feat_label_loss", type=str, default="iokr_fps__count")
    arg_parser.add_argument("--mol_feat_retention_order", type=str, default="substructure_count")

    arg_parser.add_argument("--ms2scorer", type=str, default="MetFrag_2.4.5__8afe4a14")
    arg_parser.add_argument("--molecule_identifier", type=str, default="inchikey1",
                            choices=["inchikey1", "inchi", "inchi2D"])
    arg_parser.add_argument("--max_n_candidates_train", type=int, default=50)
    arg_parser.add_argument("--max_n_candidates_test", type=int, default=50)
    arg_parser.add_argument("--L_min_train", type=int, default=4)
    arg_parser.add_argument("--L_max_train", type=int, default=32)
    arg_parser.add_argument("--L_min_test", type=int, default=50)
    arg_parser.add_argument("--L_max_test", type=int, default=75)

    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=16)
    arg_parser.add_argument("--n_trees_for_training", type=int, default=1)

    return arg_parser.parse_args()


def train_and_score(parameter_name: str, parameter_value: str):
    # ===================================
    # Set parameters to the studied value
    # ===================================
    if parameter_name == "label_loss":
        if parameter_value == "binary_tanimoto":
            args.mol_feat_label_loss = "iokr_fps__positive"
            args.label_loss = "tanimoto_loss"
        elif parameter_value == "count_minmax":
            args.mol_feat_label_loss = "iokr_fps__count"
            args.label_loss = "minmax_loss"
        elif parameter_value == "hamming":
            args.mol_feat_label_loss = "iokr_fps__positive"
            args.label_loss = "hamming"
        else:
            raise ValueError("Invalid parameter value for '%s': '%s'." % (parameter_name, parameter_value))
    elif parameter_name == "batch_size":
        args.batch_size = int(parameter_value)
    elif parameter_name == "n_init_per_example":
        args.n_init_per_example = int(parameter_value)
    elif parameter_name == "max_n_candidates_train":
        args.max_n_candidates_train = int(parameter_value)
    elif parameter_name == "step_size_approach":
        args.step_size_approach = parameter_value
    else:
        raise ValueError("Invalid parameter name: '%s'." % parameter_name)

    # ===================
    # Sequence Sample
    # ===================
    candidates = RandomSubsetCandidateSQLiteDB(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs_cand,
        number_of_candidates=args.max_n_candidates_train, include_correct_candidate=True, init_with_open_db_conn=False)
    candidates_test = RandomSubsetCandidateSQLiteDB(
        db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=rs_cand,
        number_of_candidates=args.max_n_candidates_test, include_correct_candidate=True, init_with_open_db_conn=False)

    seq_sample = SequenceSample(
        spectra, labels, candidates=candidates, N=args.n_samples_train, L_min=args.L_min_train, L_max=args.L_max_train,
        random_state=rs_seqspl, ms2scorer=args.ms2scorer)

    seq_sample_train, seq_sample_test = seq_sample.get_train_test_split(
        spectra_cv=GroupShuffleSplit(random_state=rs_gss, test_size=0.33),  # 33% of the spectra a reserved for testing
        N_train=args.n_samples_train, N_test=args.n_samples_test, L_min_test=args.L_min_test,
        L_max_test=args.L_max_test, candidates_test=candidates_test)  # type: SequenceSample, SequenceSample

    Ls_train = [len(seq) for seq in seq_sample_train]
    Ls_test = [len(seq) for seq in seq_sample_test]

    LOGGER.info("Training sequences length: min = %d, max = %d, median = %d" %
                (min(Ls_train), max(Ls_train), np.median(Ls_train).item()))
    LOGGER.info("Test sequences length: min = %d, max = %d, median = %d" %
                (min(Ls_test), max(Ls_test), np.median(Ls_test).item()))

    # ===================
    # Setup a SSVM
    # ===================
    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss=args.mol_feat_label_loss, mol_feat_retention_order=args.mol_feat_retention_order,
        mol_kernel=args.mol_kernel, C=args.C, step_size_approach=args.step_size_approach, batch_size=args.batch_size,
        n_epochs=args.n_epochs, label_loss=args.label_loss, random_state=rs_ssvm, n_jobs=4)

    # ==============
    # Train the SSVM
    # ==============
    odir = os.path.join(args.output_dir, "parameter_study", parameter_name, parameter_value)
    os.makedirs(odir, exist_ok=True)
    summary_writer = tf.summary.create_file_writer(odir)

    ssvm.fit(seq_sample_train, n_init_per_example=4, summary_writer=summary_writer, validation_data=seq_sample_test)

    # ===================
    # Score test sequence
    # ===================
    scores = ssvm.score(seq_sample_test, n_trees_per_sequence=args.n_trees_for_scoring, stype="topk_mm",
                        topk_method="csi", average=False, return_percentage=False)

    out = []
    for i, seq in enumerate(seq_sample_test):
        for k in [1, 5, 10, 20]:
            out.append([i, len(seq), seq.get_dataset(), k, scores[i, k - 1]])
    pd.DataFrame(out, columns=["sample_id", "sequence_length", "dataset", "k", "n_top_k"]) \
        .to_csv(os.path.join(odir, "topk_mm.tsv"), index=False, sep="\t")

    with open(os.path.join(odir, "parameters_.tsv"), "w+") as ofile:
        for k, v in args.__dict__.items():
            ofile.write("{} = {}\n".format(k, v))


if __name__ == "__main__":
    args = get_cli_arguments()

    # Random states
    rs_ssvm = 1993
    rs_cand = 391
    rs_seqspl = 25
    rs_gss = 103

    # ===================
    # Get list of Spectra
    # ===================
    spectra, labels = load_spectra_and_labels(args.db_fn, args.molecule_identifier)

    # ===============
    # Train and score
    # ===============
    for idx, value in enumerate(args.parameter_grid):
        LOGGER.info("Parameter: %s=%s %d/%d" % (args.parameter_to_study, value, idx + 1, len(args.parameter_grid)))
        train_and_score(args.parameter_to_study, value)



