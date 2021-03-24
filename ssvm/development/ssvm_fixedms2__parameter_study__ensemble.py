import os
import sqlite3
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from joblib.parallel import Parallel, delayed
from typing import List, Tuple
from matchms.Spectrum import Spectrum
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample, SpanningTrees
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
    arg_parser.add_argument("--n_init_per_example", type=int, default=1)
    arg_parser.add_argument("--step_size_approach", type=str, default="linesearch_parallel")
    arg_parser.add_argument("--mol_kernel", type=str, default="minmax", choices=["minmax_numba", "minmax"])
    arg_parser.add_argument("--C", type=float, default=4)
    arg_parser.add_argument("--label_loss", type=str, default="tanimoto_loss")
    arg_parser.add_argument("--mol_feat_label_loss", type=str, default="iokr_fps__positive")
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

    arg_parser.add_argument("--n_trees_for_scoring", type=int, default=1)
    arg_parser.add_argument("--n_trees_for_training", type=int, default=16)

    arg_parser.add_argument("--n_jobs", type=int, default=4)

    arg_parser.add_argument("--ssvm_update_direction", type=str, default="map")
    arg_parser.add_argument("--average_node_and_edge_potentials", type=int, default=0)
    arg_parser.add_argument("--log_transform_node_potentials", type=int, default=1)

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
    elif parameter_name == "C":
        args.C = float(parameter_value)
    elif parameter_name == "n_epochs":
        args.n_epochs = int(parameter_value)
    elif parameter_name == "n_trees_for_scoring":
        args.n_trees_for_scoring = int(parameter_value)
    elif parameter_name == "n_trees_for_training":
        args.n_trees_for_training = int(parameter_value)
    elif parameter_name == "L_train":
        args.L_min_train = int(parameter_value)
        args.L_max_train = int(parameter_value)
    elif parameter_name == "n_samples_train":
        args.n_samples_train = int(parameter_value)
    elif parameter_name == "ssvm_update_direction":
        args.ssvm_update_direction = parameter_value
    elif parameter_name == "potential_options":
        if parameter_value == "no_avg__no_log":
            args.average_node_and_edge_potentials = False
            args.log_transform_node_potentials = False
        elif parameter_value == "no_avg__log":
            args.average_node_and_edge_potentials = False
            args.log_transform_node_potentials = True
        elif parameter_value == "avg__no_log":
            args.average_node_and_edge_potentials = True
            args.log_transform_node_potentials = False
        elif parameter_value == "avg__log":
            args.average_node_and_edge_potentials = True
            args.log_transform_node_potentials = True
        else:
            raise ValueError("Invalid parameter value for '%s': '%s'." % (parameter_name, parameter_value))
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
    l_ssvm = [
        StructuredSVMSequencesFixedMS2(
            mol_feat_label_loss=args.mol_feat_label_loss, mol_feat_retention_order=args.mol_feat_retention_order,
            mol_kernel=args.mol_kernel, C=args.C, step_size_approach=args.step_size_approach,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs, label_loss=args.label_loss, random_state=rs_ssvm + idx, n_jobs=args.n_jobs,
            update_direction=args.ssvm_update_direction,
            log_transform_node_potentials=args.log_transform_node_potentials,
            average_node_and_edge_potentials=args.average_node_and_edge_potentials, n_trees_per_sequence=1)
        for idx in range(args.n_trees_for_training)
    ]

    # ==============
    # Train the SSVM
    # ==============
    odir = os.path.join(args.output_dir, "parameter_study", parameter_name, parameter_value)
    for idx in range(args.n_trees_for_training):
        os.makedirs(os.path.join(odir, "tree_idx=%d" % idx), exist_ok=True)
        summary_writer = tf.summary.create_file_writer(os.path.join(odir, "tree_idx=%d" % idx))

        l_ssvm[idx].fit(
            seq_sample_train, n_init_per_example=args.n_init_per_example,
            summary_writer=None, validation_data=None
        )

    # ===================
    # Predict max-margins
    # ===================
    l_marg = []
    for idx in range(args.n_trees_for_training):
        # Get spanning trees for the test examples (its a different one for each SSVM model)
        l_Gs = [
            SpanningTrees(seq, n_trees=args.n_trees_for_scoring, random_state=rs_score + idx) for seq in seq_sample_test
        ]

        # Calculate the max-marginals
        l_marg.append(
            Parallel(n_jobs=args.n_jobs)(
                delayed(l_ssvm[idx].predict)(seq, Gs=Gs, map=False, n_jobs=1) for seq, Gs in zip(seq_sample_test, l_Gs)
            )
        )

    # ===================
    # Score test sequence
    # ===================
    out_ind = []
    out_avg = []
    for i, seq in enumerate(seq_sample_test):
        # Performance per SSVM
        for idx in range(args.n_trees_for_training):
            _scores = l_ssvm[0]._topk_score(seq, l_marg[idx][i], False, 20, True, "csi")
            for k in [1, 5, 10, 20]:
                out_ind.append([i, len(seq), seq.get_dataset(), k, _scores[k - 1], idx])

        # Average the margins
        for n_marg in range(1, args.n_trees_for_training + 1):
            _avg_marg = l_marg[0][i]  # type: dict
            for idx in range(1, n_marg):
                for node in _avg_marg:
                    assert _avg_marg[node]["n_cand"] == l_marg[idx][i][node]["n_cand"]
                    _avg_marg[node]["score"] += l_marg[idx][i][node]["score"]

            for node in _avg_marg:
                _avg_marg[node]["score"] /= n_marg

            _scores = l_ssvm[0]._topk_score(seq, _avg_marg, False, 20, True, "csi")
            for k in [1, 5, 10, 20]:
                out_avg.append([i, len(seq), seq.get_dataset(), k, _scores[k - 1], n_marg])

    pd.DataFrame(out_ind, columns=["sample_id", "sequence_length", "dataset", "k", "n_top_k", "tree_index"]) \
        .to_csv(os.path.join(odir, "topk_mm__ind.tsv"), index=False, sep="\t")

    pd.DataFrame(out_avg, columns=["sample_id", "sequence_length", "dataset", "k", "n_top_k", "n_trees"]) \
        .to_csv(os.path.join(odir, "topk_mm__avg.tsv"), index=False, sep="\t")

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
    rs_score = 2942

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



