import sqlite3
import os
import pandas as pd
import numpy as np
import tensorflow as tf

from matchms.Spectrum import Spectrum

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

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
        mol_kernel="minmax", C=2, step_size="linesearch", batch_size=16, n_epochs=1, label_loss="tanimoto_loss",
        random_state=1993, retention_order_weight=1.0, n_jobs=4)

    N = 100
    seq_sample = SequenceSample(
        spectra, labels,
        RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey1", random_state=2,
                                      number_of_candidates=50, include_correct_candidate=True,
                                      init_with_open_db_conn=False),
        N=N, L_min=10,
        L_max=20, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

    seq_sample_train, seq_sample_test = seq_sample.get_train_test_split(cv=4)
    print(len(seq_sample_train[0]))

    summary_writer = tf.summary.create_file_writer(os.path.join(tf_summary_base_dir, "test_performance",
                                                                "%d" % np.random.randint(1000)))

    ssvm.fit(seq_sample_train, n_init_per_example=3, summary_writer=None)

    print("SCORING")
    print(ssvm.score(seq_sample_test, stype="top1_map"))
