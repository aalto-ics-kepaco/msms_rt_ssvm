import sqlite3
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from typing import List, Tuple

from matchms.Spectrum import Spectrum
from sklearn.model_selection import GroupShuffleSplit

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2


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


if __name__ == "__main__":
    molecule_identifier = "inchikey1"
    N_train = 250
    N_test = 15

    # Random states
    rs_ssvm = 1993
    rs_cand = 391
    rs_seqspl = 25
    rs_gss = 103

    tf_summary_base_dir = "/home/bach/Documents/doctoral/projects/msms_rt_ssvm/ssvm/development/logs"
    DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"

    # ===================
    # Get list of Spectra
    # ===================
    spectra, labels = load_spectra_and_labels(DB_FN, molecule_identifier)

    # ===================
    # Setup a SSVM
    # ===================
    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss="iokr_fps__count", mol_feat_retention_order="substructure_count",
        mol_kernel="minmax", C=16, step_size="linesearch_parallel", batch_size=16, n_epochs=5, label_loss="minmax_loss",
        random_state=rs_ssvm, n_jobs=4)

    # ===================
    # Sequence Sample
    # ===================
    candidates = RandomSubsetCandidateSQLiteDB(
        db_fn=DB_FN, molecule_identifier=molecule_identifier, random_state=rs_cand, number_of_candidates=50,
        include_correct_candidate=True, init_with_open_db_conn=False)

    seq_sample = SequenceSample(
        spectra, labels, candidates=candidates, N=N_train, L_min=6, L_max=32, random_state=rs_seqspl,
        ms2scorer="MetFrag_2.4.5__8afe4a14")

    seq_sample_train, seq_sample_test = seq_sample.get_train_test_split(
        spectra_cv=GroupShuffleSplit(random_state=rs_gss, test_size=0.33),  # 33% of the spectra a reserved for testing
        N_train=N_train, N_test=N_test, L_min_test=50, L_max_test=75)

    # ==============
    # Train the SSVM
    # ==============
    summary_writer = tf.summary.create_file_writer(os.path.join(tf_summary_base_dir, "parameter_exploration",
                                                                "%d" % np.random.randint(1000)))

    ssvm.fit(seq_sample_train, n_init_per_example=6, summary_writer=summary_writer)

    print("SCORING")
    print(ssvm.score(seq_sample_test, stype="top1_map"))
