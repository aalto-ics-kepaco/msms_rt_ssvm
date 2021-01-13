import sqlite3
import pandas as pd
import numpy as np
import time

from matchms.Spectrum import Spectrum

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2


def run(mol_kernel, seq_sample, n_jobs):
    # ===================
    # Setup a SSVM
    # ===================
    ssvm = StructuredSVMSequencesFixedMS2(
        mol_feat_label_loss="iokr_fps__positive", mol_feat_retention_order="substructure_count",
        mol_kernel=mol_kernel, C=2, step_size="linesearch", batch_size=8, n_epochs=3, label_loss="tanimoto_loss",
        random_state=1993, retention_order_weight=0.5, n_jobs=n_jobs)

    ssvm.fit(seq_sample, n_init_per_example=5)


if __name__ == "__main__":
    # DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"
    DB_FN = "/home/bach/tmp/DB_LATEST.db"

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

    N = 24
    seq_sample = SequenceSample(
        spectra, labels,
        RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchi", random_state=192,
                                      number_of_candidates=50, include_correct_candidate=True,
                                      init_with_open_db_conn=False),
        N=N, L_min=10,
        L_max=15, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

    n_rep = 3
    t_np = 0.0
    t_np_mc = 0.0
    t_numba = 0.0

    print("Fill cache")
    run("minmax", seq_sample, 1)

    print("Numpy")
    for _ in range(n_rep):
        t = time.time()
        run("minmax", seq_sample, 1)
        t_np += (time.time() - t)

    print("Numpy (multicore)")
    for _ in range(n_rep):
        t = time.time()
        run("minmax", seq_sample, 4)
        t_np_mc += (time.time() - t)

    # print("Numba")
    # for _ in range(n_rep):
    #     t = time.time()
    #     run("minmax_numba", seq_sample, 1)
    #     t_numba += (time.time() - t)

    print("Numpy: %.3fs" % (t_np / n_rep))
    print("Numpy (multicore): %.3fs" % (t_np_mc / n_rep))
    print("Numba: %.3fs" % (t_numba / n_rep))

    # TODO: We somehow should ensure that the database connection is always closed.
    # seq_sample.candidates.close()
