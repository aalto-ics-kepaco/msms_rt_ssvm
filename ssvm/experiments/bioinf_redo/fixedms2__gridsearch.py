import sqlite3
import os
import pandas as pd
import numpy as np
import itertools as it
import argparse
import pickle
import gzip
import logging
import time
import sys

from typing import Tuple
from matchms.Spectrum import Spectrum
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, FixedSubsetCandidateSQLiteDB, CandidateSQLiteDB
from ssvm.data_structures import LabeledSequence, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

from msmsrt_scorer.lib.data_utils import load_dataset_CASMI, load_dataset_EA
from msmsrt_scorer.experiments.plot_and_table_utils import _marg_or_cand

# ================
# Setup the Logger
LOGGER = logging.getLogger("redo_bioinf_exp")
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

CH = logging.StreamHandler()
CH.setLevel(logging.INFO)

FORMATTER = logging.Formatter('[%(levelname)s] %(name)s : %(message)s')
CH.setFormatter(FORMATTER)

LOGGER.addHandler(CH)
# ================


def get_cli_arguments() -> argparse.Namespace:
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
    arg_parser.add_argument("--output_dir", type=str,
                            help="Base directory to store the Tensorboard logging files, train and test splits, ...",
                            default="./logs/fixedms2")
    arg_parser.add_argument("--n_samples_train", type=int, default=1000,
                            help="Number of training sample sequences.")
    arg_parser.add_argument("--n_epochs", type=int, default=7)
    arg_parser.add_argument("--batch_size", type=int, default=8)
    arg_parser.add_argument("--n_init_per_example", type=int, default=6)
    arg_parser.add_argument("--max_n_train_candidates", type=int, default=100)
    arg_parser.add_argument("--stepsize", type=str, default="linesearch")
    arg_parser.add_argument("--ms2scorer", type=str, default="MetFrag_2.4.5__8afe4a14")
    arg_parser.add_argument("--mol_kernel", type=str, default="minmax_numba", choices=["minmax_numba", "minmax"])
    arg_parser.add_argument("--molecule_identifier", type=str, default="inchi2D", choices=["inchikey1", "inchi", "inchi2D"])
    arg_parser.add_argument("--lloss_fps_mode", type=str, default="binary", choices=["binary", "count"])

    return arg_parser.parse_args()


def get_hparam_estimation_setting(debug: bool) -> Tuple:
    if args.debug:
        n_splits_inner = 2
        n_epochs_inner = 1
        C_grid = [1, 64]
        rtw_grid = [0.5]
    else:
        n_splits_inner = 10
        n_epochs_inner = 5
        C_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        rtw_grid = [0.3, 0.4, 0.5, 0.6, 0.7]

    return n_splits_inner, n_epochs_inner, C_grid, rtw_grid


if __name__ == "__main__":
    # ===================
    # Parse arguments
    # ===================
    args = get_cli_arguments()
    LOGGER.info("=== Arguments ===")
    for k, v in args.__dict__.items():
        LOGGER.info("{} = {}".format(k, v))

    # Handle parameters regarding the label-loss
    if args.lloss_fps_mode == "count":
        mol_feat_label_loss = "iokr_fps__count"
        label_loss = "minmax_loss"
    else:
        mol_feat_label_loss = "iokr_fps__positive"
        label_loss = "tanimoto_loss"

    n_splits_inner, n_epochs_inner, C_grid, rtw_grid = get_hparam_estimation_setting(args.debug)

    # ==================================
    # Get the evaluation dataset from ID
    # ==================================
    if args.eval_set_id in range(0, 50):
        eval_ds = "CASMI"
        eval_ion = "negative"
        eval_max_n_ms2 = 50
        eval_idx = args.eval_set_id
    elif args.eval_set_id in range(50, 100):
        eval_ds = "CASMI"
        eval_ion = "positive"
        eval_max_n_ms2 = 75
        eval_idx = args.eval_set_id - 50
    elif args.eval_set_id in range(100, 150):
        eval_ds = "EA"
        eval_ion = "negative"
        eval_max_n_ms2 = 65
        eval_idx = args.eval_set_id - 100
    else:  # in range(150, 250)
        eval_ds = "EA"
        eval_ion = "positive"
        eval_max_n_ms2 = 100
        eval_idx = args.eval_set_id - 150

    LOGGER.info("=== Dataset ===")
    LOGGER.info("DS = %s, ION = %s" % (eval_ds, eval_ion))

    # ===================
    # Get list of Spectra
    # ===================
    db = sqlite3.connect("file:" + args.db_fn + "?mode=ro", uri=True)

    # Read in spectra and labels
    res = pd.read_sql_query("SELECT spectrum, %s as molecule, rt, challenge FROM challenges_spectra "
                            "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum"
                            "   INNER JOIN molecules m on m.inchi = s.molecule" % args.molecule_identifier, con=db)
    spectra = [Spectrum(np.array([]), np.array([]),
                        {"spectrum_id": spec_id, "retention_time": rtw, "dataset": chlg, "molecule_id": mol})
               for (spec_id, rtw, chlg, mol) in zip(res["spectrum"], res["rt"], res["challenge"], res["molecule"])]
    labels = res["molecule"].to_list()

    # ========================
    # Get test set spectra IDs
    # ========================
    if eval_ds == "CASMI":
        challenges, candidates = load_dataset_CASMI(db, ion_mode=eval_ion, participant=args.ms2scorer,
                                                    prefmodel="c6d6f521", sort_candidates_by_ms2_score=False,
                                                    restrict_candidates_to_correct_mf=False)
    else:
        challenges, candidates = load_dataset_EA(db, ion_mode=eval_ion, participant=args.ms2scorer, sample_idx=eval_idx,
                                                 sort_candidates_by_ms2_score=False,
                                                 prefmodel={"training_dataset": "MEOH_AND_CASMI_JOINT",
                                                            "keep_test_molecules": False,
                                                            "estimator": "ranksvm",
                                                            "molecule_representation": "substructure_count"})

    with gzip.open(os.path.join(
            "split_data", eval_ds, eval_ion,
            _marg_or_cand("candidates", max_n_ms2=eval_max_n_ms2, sample_id=eval_idx))) as eval_set_file:
        eval_set = pickle.load(eval_set_file)["test_set"]

    spec_ids_eval = [challenges[i]["name"] for i in eval_set]
    spectra_eval, labels_eval = zip(*[(spectrum, label) for spectrum, label in zip(spectra, labels)
                                      if spectrum.get("spectrum_id") in spec_ids_eval])

    labelspace_subset = {challenges[i]["name"]: candidates[i]["structure"] for i in eval_set}

    db.close()

    # =================================
    # Find the optimal hyper-parameters
    # =================================
    spectra_train, labels_train = zip(*[(spectrum, label) for spectrum, label in zip(spectra, labels)
                                        if label not in labels_eval])

    LOGGER.info("Number of spectra: total = %d, train = %d, evaluation = %d"
                % (len(spectra), len(spectra_train), len(spectra_eval)))

    param_grid = list(it.product(C_grid, rtw_grid))
    opt_values = []
    cv_inner = GroupShuffleSplit(n_splits=n_splits_inner, test_size=0.2, random_state=102)
    LOGGER.info("=== Search hyper parameter grid ===")
    LOGGER.info("C-grid: {}".format(C_grid))
    LOGGER.info("RTW-grid: {}".format(rtw_grid))
    for idx, (C, rtw) in enumerate(param_grid):
        LOGGER.info("(%d / %d) C = %f, rtw = %f" % (idx + 1, len(param_grid), C, rtw))

        for jdx, (train, test) in enumerate(cv_inner.split(spectra_train, groups=labels_train)):
            LOGGER.info("(%d / %d, inner) Number of spectra (inner cv): train = %d, test = %d"
                        % (jdx + 1, n_splits_inner, len(train), len(test)))

            spectra_train_inner = [spectra_train[i] for i in train]
            labels_train_inner = [labels_train[i] for i in train]

            # Get training sequences
            # ----------------------
            training_sequences = SequenceSample(
                spectra_train_inner, labels_train_inner,
                RandomSubsetCandidateSQLiteDB(
                    db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=jdx,
                    number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
                    init_with_open_db_conn=False),
                N=np.int(np.round(args.n_samples_train * 0.75)), L_min=10, L_max=30, random_state=jdx,
                ms2scorer=args.ms2scorer)

            # Train the SSVM
            # --------------
            ssvm = StructuredSVMSequencesFixedMS2(
                mol_feat_label_loss=mol_feat_label_loss, mol_feat_retention_order="substructure_count",
                mol_kernel=args.mol_kernel, C=C, step_size_approach=args.stepsize, batch_size=args.batch_size,
                n_epochs=n_epochs_inner, label_loss=label_loss, random_state=jdx,
                retention_order_weight=rtw, n_jobs=args.n_jobs
            ).fit(training_sequences, n_init_per_example=args.n_init_per_example)

            # Access test set performance
            # ---------------------------
            spectra_test_inner = [spectra_train[i] for i in test]
            labels_test_inner = [labels_train[i] for i in test]

            test_sequences = SequenceSample(
                spectra_test_inner, labels_test_inner,
                RandomSubsetCandidateSQLiteDB(
                    db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=jdx,
                    number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
                    init_with_open_db_conn=False),
                N=np.int(np.round(args.n_samples_train * 0.25)), L_min=10, L_max=30, random_state=jdx,
                ms2scorer=args.ms2scorer)

            LOGGER.info("Score hyper-parameter tuple ...")
            opt_values.append([C, rtw, jdx, ssvm.score(test_sequences, stype="ndcg_ohc")])

    df_opt_values = pd.DataFrame(opt_values, columns=["C", "retention_order_weight", "inner_split", "ndcg_ohc"])
    C_opt, rtw_opt = df_opt_values.groupby(["C", "retention_order_weight"]).mean()["ndcg_ohc"].idxmax()

    LOGGER.info("C_opt=%f, rtw_opt=%f" % (C_opt, rtw_opt))
    LOGGER.info("C-grid: {}".format(C_grid))
    LOGGER.info(df_opt_values.groupby(["C"]).mean()["ndcg_ohc"].values)
    LOGGER.info("RTW-grid: {}".format(rtw_grid))
    LOGGER.info(df_opt_values.groupby(["retention_order_weight"]).mean()["ndcg_ohc"].values)

    # =========================================
    # Train the SSVM with best hyper-parameters
    # =========================================
    LOGGER.info("=== Train SSVM with all training data ===")
    training_sequences = SequenceSample(
        spectra_train, labels_train,
        RandomSubsetCandidateSQLiteDB(
            db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, random_state=425,
            number_of_candidates=args.max_n_train_candidates, include_correct_candidate=True,
            init_with_open_db_conn=False),
        N=args.n_samples_train, L_min=10, L_max=30, random_state=23,
        ms2scorer=args.ms2scorer)

    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss=mol_feat_label_loss, mol_feat_retention_order="substructure_count",
        mol_kernel=args.mol_kernel, C=C_opt, step_size_approach=args.stepsize, batch_size=args.batch_size,
        n_epochs=args.n_epochs, label_loss=label_loss, random_state=1993,
        retention_order_weight=rtw_opt, n_jobs=args.n_jobs
    ).fit(training_sequences, n_init_per_example=args.n_init_per_example)

    # ====================
    # Evaluate performance
    # ====================
    LOGGER.info("=== Evaluate SSVM performance on the evaluation set ===")
    candidates = FixedSubsetCandidateSQLiteDB(
        labelspace_subset=labelspace_subset, db_fn=args.db_fn, molecule_identifier=args.molecule_identifier,
        init_with_open_db_conn=False, assert_correct_candidate=True)

    # candidates = CandidateSQLiteDB(
    #     db_fn=args.db_fn, molecule_identifier=args.molecule_identifier, init_with_open_db_conn=False)

    with candidates:
        LOGGER.info("\tSpectrum - n_candidates:")
        for spec in spectra_eval:
            LOGGER.info("\t%s - %5d" % (spec.get("spectrum_id"), len(candidates.get_labelspace(spec))))

    seq_eval = LabeledSequence(spectra_eval, labels_eval, candidates=candidates, ms2scorer=args.ms2scorer)

    scores = {}
    for stype in ["topk_mm", "top1_map"]:
        LOGGER.info("Evaluation set scoring: %s" % stype)
        start = time.time()
        scores[stype] = ssvm.score([seq_eval], stype=stype, average=False, n_trees_per_sequence=4).flatten()
        LOGGER.info("Scoring time (%s): %.3fs" % (stype, time.time() - start))

    LOGGER.info("topk (raw): {}".format(scores["topk_mm"]))
    LOGGER.info("Top-1 (MAP) = %.2f%%" % scores["top1_map"])
    LOGGER.info("Top-1 (MM) = %.2f%%, Top-5 (MM) = %.2f%%, Top-10 (MM) = %.2f%%, Top-20 (MM) = %.2f%%"
                % (scores["topk_mm"][0], scores["topk_mm"][4], scores["topk_mm"][9], scores["topk_mm"][19]))

    # =================
    # Write out results
    # =================
    LOGGER.info("=== Write out results ===")
    odir_res = os.path.join(args.output_dir, "ds=%s__ion=%s" % (eval_ds, eval_ion))
    os.makedirs(odir_res, exist_ok=True)
    LOGGER.info("Output directory: %s" % odir_res)

    df_opt_values.to_csv(os.path.join(odir_res, "grid_search_results__spl=%03d.tsv" % eval_idx), sep="\t", index=False)

    pd.DataFrame(zip(range(1, len(scores["topk_mm"]) + 1), scores["topk_mm"]), columns=["k", "top_acc_perc"]) \
        .to_csv(os.path.join(odir_res, "topk__spl=%03d.tsv" % eval_idx), sep="\t", index=False)

    with open(os.path.join(odir_res, "top1_map__spl=%03d.tsv" % eval_idx), "w+") as ofile:
        ofile.write("%f" % scores["top1_map"])

    with open(os.path.join(odir_res, "eval_spec_ids__spl=%03d.tsv" % eval_idx), "w+") as ofile:
        ofile.write("\n".join(spec_ids_eval))

    sys.exit(0)
