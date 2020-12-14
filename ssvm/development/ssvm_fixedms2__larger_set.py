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
    res = pd.read_sql_query("SELECT spectrum, molecule, rt, challenge FROM challenges_spectra "
                            "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum", con=db)
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
        mol_kernel="minmax_fast", C=2, step_size="linesearch", batch_size=8, n_epochs=7, label_loss="tanimoto_loss",
        random_state=1993, retention_order_weight=0.5)

    N = 500
    seq_sample = SequenceSample(
        spectra, labels,
        RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchi", random_state=2,
                                      number_of_candidates=50, include_correct_candidate=True),
        N=N, L_min=10,
        L_max=20, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

    summary_writer = tf.summary.create_file_writer(os.path.join(tf_summary_base_dir, "larger_set",
                                                                "%d" % np.random.randint(1000)))

    ssvm.fit(seq_sample, n_init_per_example=5, summary_writer=None)

    # TODO: We somehow should ensure that the database connection is always closed.
    seq_sample.candidates.close()

