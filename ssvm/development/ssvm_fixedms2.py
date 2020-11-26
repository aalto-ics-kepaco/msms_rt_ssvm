import sqlite3
import pandas as pd
import numpy as np

from matchms.Spectrum import Spectrum

from ssvm.data_structures import RandomSubsetCandidateSQLiteDB, SequenceSample
from ssvm.ssvm import StructuredSVMSequencesFixedMS2

if __name__ == "__main__":
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
        mol_kernel="minmax", C=2, step_size="linesearch", batch_size=2)

    N = 50
    seq_sample = SequenceSample(
        spectra, labels,
        RandomSubsetCandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchi", random_state=192,
                                      number_of_candidates=50, include_correct_candidate=True),
        N=N, L_min=10,
        L_max=15, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

    ssvm.fit(seq_sample, n_init_per_example=1)

