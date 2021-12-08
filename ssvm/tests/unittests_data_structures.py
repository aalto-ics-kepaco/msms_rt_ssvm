####
#
# The MIT License (MIT)
#
# Copyright 2020, 2021 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####
import sqlite3
import unittest
import pandas as pd
import numpy as np
import itertools as it
import networkx as nx
import time

from matchms.Spectrum import Spectrum
from joblib import Parallel, delayed
from scipy.stats import rankdata

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from ssvm.data_structures import ABCCandSQLiteDB
from ssvm.data_structures import CandSQLiteDB_Bach2020, RandomSubsetCandSQLiteDB_Bach2020
from ssvm.data_structures import CandSQLiteDB_Massbank, RandomSubsetCandSQLiteDB_Massbank
from ssvm.data_structures import ImputationError
from ssvm.data_structures import SequenceSample, Sequence, SpanningTrees, LabeledSequence
from ssvm.loss_functions import zeroone_loss

BACH2020_DB_FN = "Bach2020_test_db.sqlite"
MASSBANK_DB_FN = "Massbank_test_db.sqlite"


class TestMassbankCandidateSQLiteDB(unittest.TestCase):
    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "LU43956814"})

        # Molecule identifier: inchikey
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="cid")
        self.assertEqual(231, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(84, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952"})

        # Molecule identifier: inchikey
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="cid")
        self.assertEqual(2308, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1923, candidates.get_n_cand(spectrum))

    def test_get_constant_ms_score(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "AU88550178"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        scores = candidates.get_ms_scores(spectrum, ms_scorer="CONST_MS_SCORE")
        self.assertEqual(3934, len(scores))
        self.assertTrue(np.all(np.array(scores) == 1.0))

    def test_get_multiple_ms2_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------

        # - MetFrag has not scored that spectrum
        # - SIRIUS scores are available

        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "LQB6372613"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        for s2w in [0, 0.32, 0.5, 0.72, 1]:
            scores = candidates.get_ms_scores(
                spectrum, ms_scorer=["sirius__sd__correct_mf", "metfrag__norm_after_merge"],
                ms_scorer_weights=[1 - s2w, s2w]
            )

            self.assertEqual(118, len(scores))
            self.assertTrue(np.all(np.array(scores) > 0))
            self.assertEqual(1.0, np.max(scores))

            _scores_sirius = candidates.get_ms_scores(
                spectrum, ms_scorer="sirius__sd__correct_mf", return_as_ndarray=True
            )
            np.testing.assert_equal(((1 - s2w) * _scores_sirius + s2w), scores)

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BML0302194"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        for s2w in [0, 0.32, 0.5, 0.72, 1]:
            scores = candidates.get_ms_scores(
                spectrum, ms_scorer=["sirius__sd__correct_mf", "metfrag__norm_after_merge"],
                ms_scorer_weights=[1 - s2w, s2w]
            )

            self.assertEqual(1271, len(scores))
            self.assertTrue(np.all(np.array(scores) > 0))
            self.assertEqual(1.0, np.max(scores))

            _scores_sirius = candidates.get_ms_scores(
                spectrum, ms_scorer="sirius__sd__correct_mf", return_as_ndarray=True
            )
            _scores_metfrag = candidates.get_ms_scores(
                spectrum, ms_scorer="metfrag__norm_after_merge", return_as_ndarray=True
            )

            _scores_comb = ((1 - s2w) * _scores_sirius + s2w * _scores_metfrag)
            _scores_comb /= np.max(_scores_comb)

            np.testing.assert_equal(_scores_comb, scores)

    def test_get_ms2_scores__normalized(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "AU88550178"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf")
        self.assertEqual(3934, len(scores))
        self.assertTrue(np.all(np.array(scores) > 0))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(3934, len(scores))
        self.assertEqual(0.0627934092603395, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf")
        self.assertEqual(2308, len(scores))
        self.assertTrue(np.all(np.array(scores) > 0))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(2308, len(scores))
        self.assertEqual(0.11534319471384565, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # ----------
        # SPECTRUM 3
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "LQB6372613"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf")
        self.assertEqual(118, len(scores))
        self.assertTrue(np.all(np.array(scores) > 0))
        self.assertEqual(1.0, np.max(scores))

        # # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(118, len(scores))
        self.assertEqual(1.0, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

    def test_get_ms2_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "AU88550178"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf", scale_scores_to_range=False)
        self.assertEqual(3934, len(scores))
        self.assertEqual(-516.7701408961747, np.min(scores))
        self.assertEqual(-60.65141796101968, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge", scale_scores_to_range=False)
        self.assertEqual(3934, len(scores))
        self.assertEqual(59.08480943917096, np.min(scores))
        self.assertEqual(940.939664451204, np.max(scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf", scale_scores_to_range=False)
        self.assertEqual(2308, len(scores))
        self.assertEqual(-616.7732484911674, np.min(scores))
        self.assertEqual(-138.34323578753632, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge", scale_scores_to_range=False)
        self.assertEqual(2308, len(scores))
        self.assertEqual(111.97300932463722, np.min(scores))
        self.assertEqual(970.7812377005032, np.max(scores))

        # ----------
        # SPECTRUM 3
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "LQB6372613"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is SIRIUS
        scores = candidates.get_ms_scores(spectrum, ms_scorer="sirius__sd__correct_mf", scale_scores_to_range=False)
        self.assertEqual(118, len(scores))
        self.assertEqual(-307.381287577836, np.min(scores))
        self.assertEqual(-147.8959133560828, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge", scale_scores_to_range=False)
        self.assertEqual(118, len(scores))
        self.assertEqual(1e-6, np.min(scores))
        self.assertEqual(1e-6, np.max(scores))

    def test_get_molecule_features(self):
        # SIRIUS Fingerprints
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "PR37531286"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")

        fps = candidates.get_molecule_features(spectrum, features="sirius_fps")
        self.assertEqual((586, 3047), fps.shape)
        self.assertTrue(np.all(np.isin(fps, [0, 1])))

        # FCFP features (encodes stereo)
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "PR37531286"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="cid")

        fps = candidates.get_molecule_features(spectrum, features="FCFP__count__all")
        self.assertEqual((743, 1280), fps.shape)
        self.assertTrue(np.all(fps >= 0))

        df = candidates.get_molecule_features(spectrum, features="FCFP__count__all", return_dataframe=True)
        self.assertEqual((743, 1 + 1280), df.shape)
        fp_i = df[df["identifier"] == 20372696].iloc[0, 1:].values

        fp_i_ref = np.zeros_like(fp_i)
        for j, v in zip(
                "0,2,3,18,51,84,87,90,95,96,287,562,565,827,863,1142".split(","),
                "9,6,1,1,4,1,1,1,1,2,1,1,1,1,1,1".split(",")
        ):
            fp_i_ref[int(j)] = int(v)

        self.assertEqual((1280, ), fp_i.shape)
        np.array_equal(fp_i, fp_i_ref)

        # Bouwmeester et al. (2019) descriptors
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "KW45044969"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")

        desc = candidates.get_molecule_features(spectrum, features="bouwmeester__smiles_can")
        self.assertEqual((339, 196), desc.shape)
        self.assertFalse(np.any(np.isnan(desc)))

        df = candidates.get_molecule_features(spectrum, features="bouwmeester__smiles_can", return_dataframe=True)
        self.assertEqual((339, 1 + 196), df.shape)

        ref_descs = [
            ("JATOIOIIXORRLF", "2.2505765195661276,588.1625722468402,12.303118619434356,8.776359180062045,9.59285576098977,8.044624757173649,4.868749138635866,6.233897510305976,3.4014670750900127,4.423595627720499,2.162972904196923,2.566041356664746,1.4551450719093095,1.809457319866485,10.399000581649606,8.417796984328938,0.0,5.749511833283905,0.0,11.126902983393991,12.13273413692322,12.13273413692322,30.33183534230805,4.183085432649707,4.552749873690364,250.0299798,0.0,-1.8499999999999996,17.0,240.19499999999996,7071.721377876014,11.63001993693029,4.565685103723021,3.3024254077918185,99.43968244116651,10.488900419365788,0.4459747048137353,10.488900419365788,0.4459747048137353,0.07609363189720364,0.3618096010624026,-4.454659260309061,-0.3618096010624026,2.5351999999999997,64.05460000000004,250.27499999999998,1.0,4.0,0.0,0.0,0.0,2.0,0.0,2.0,3.0,1.0,5.0,0.0,3.0,0.0,0.0,0.0,88.0,4.183085432649707,5.749511833283905,0.0,0.0,0.0,10.399000581649606,4.552749873690364,0.0,8.417796984328938,0.0,42.46456947923127,23.259637120317212,0.0,0.0,2.0,17.15363229066901,10.399000581649606,0.0,0.0,0.0,0.0,0.0,54.59730361615449,0.0,16.876414816677897,4.183085432649707,0.0,5.749511833283905,0.0,12.9705468580193,10.399000581649606,0.0,0.0,54.59730361615449,0.0,11.126902983393991,0.0,63.60000000000001,33.80553398601559,0.0,0.0,0.0,1.957586451247166,0.07609363189720364,16.03211185781577,0.0,0.0,-4.454659260309061,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"),
            ("ZQJJGLRMKXLYTQ", "2.250456030139938,602.7222724681443,12.250712376627058,8.881171665676636,9.697668246604362,8.16470379794645,5.0411565874769755,6.026755147130464,3.4281342285081533,4.5648840591942745,2.1829390583281376,3.1867093796204986,1.3919360981392794,2.2583498768604438,11.594891607029837,9.589074368143644,0.0,6.4208216229260096,16.231357223906805,5.386224214464796,11.761884949391115,18.19910120538483,12.13273413692322,0.0,9.523678331894054,250.0299798,0.16666666666666666,-1.7499999999999996,17.0,240.19499999999996,7784.75003983114,11.72715976331361,5.053433909859866,2.87974497096204,101.25309879607984,11.5739630574452,0.48120831865481606,11.5739630574452,0.3495474441740278,0.024735607205845334,0.3495474441740278,-0.8725213634836646,-0.48120831865481606,2.3598,65.46080000000002,250.27499999999995,1.0,4.0,0.0,0.0,0.0,1.0,1.0,2.0,4.0,1.0,5.0,0.0,4.0,0.0,0.0,0.0,88.0,9.523678331894054,5.583020141642242,0.0,0.0,0.0,11.594891607029837,4.794537184071822,4.794537184071822,0.0,11.761884949391115,18.19910120538483,12.13273413692322,11.139077821211584,11.316305098443785,2.0,14.318215515965875,28.700434593449998,0.0,0.0,0.0,11.316305098443785,5.752853606746789,40.75195884545786,0.0,0.0,5.625586319077987,0.0,0.0,11.761884949391115,16.82868628953934,4.794537184071822,0.0,6.4208216229260096,44.439006938950996,0.0,10.969244356107037,0.0,67.51,5.126489040060469,1.2031264172335598,22.39491072317867,9.355555147997878,0.12423458889728756,-0.512964248971193,8.95057939106415,0.024735607205845334,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0"),
            ("ZPBSSLBUJMLESN", "3.0586897626128566,701.7625637729037,12.629392033067417,8.79038045871581,9.606877039643535,7.947706970623633,4.795262425588228,6.235325754388867,3.5438512452502637,4.902232215370441,2.482800355462781,3.411970242978388,1.740919785352966,2.566907287907442,20.763122167835235,13.524324379169643,0.0,10.949675706161791,5.386224214464796,0.0,12.142387175295491,24.26546827384644,0.0,6.578935683598497,4.552749873690364,250.0299798,0.0,-1.8499999999999996,17.0,240.19499999999996,5998.775278840748,11.63001993693029,3.868405274120684,1.9008241027006003,98.57621861007776,11.29709939531368,0.5058345503991459,11.29709939531368,0.29836686307903043,0.2604861111111112,0.29836686307903043,-4.490462962962963,-0.5058345503991459,2.4351000000000007,65.51840000000003,250.27499999999998,2.0,4.0,0.0,0.0,0.0,2.0,0.0,2.0,3.0,2.0,5.0,0.0,2.0,0.0,0.0,0.0,88.0,5.106527394840706,10.644995308801679,0.0,0.0,10.118126859033554,0.0,4.552749873690364,0.0,8.417796984328938,0.0,36.92042406427882,11.452591282926406,10.949675706161791,0.0,2.0,18.077074252860008,26.966595394797025,0.0,0.0,0.0,4.895483475517775,0.0,42.47422251760354,0.0,5.749511833283905,0.0,0.0,5.749511833283905,0.0,18.077074252860008,10.118126859033554,0.0,5.563451491696996,41.80625450142432,0.0,16.848468535763473,0.0,74.6,31.759926303854872,0.0,-0.4810185185185183,10.692512282690855,0.2604861111111112,-0.48499999999999965,8.178457105064247,1.3298148148148148,3.4852848639455782,-4.490462962962963,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0")
        ]

        for mol_id, ref_desc in ref_descs:
            desc_i = df[df["identifier"] == mol_id].iloc[0, 1:].values
            np.array_equal(desc_i, np.fromstring(ref_desc, sep=","))

    def test_get_molecule_feature_by_id_query(self):
        molecule_ids = [
            "UKQKUSAOCWFKSE-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "TVKGYMYAOVADOP-SOFGYWHQSA-N",
            "HZINSBNIWQLTKI-UHFFFAOYSA-N",
            "UKQKUSAOCWFKSE-UHFFFAOYSA-N",
            "LQMQZNHDYCBFJK-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "TVKGYMYAOVADOP-SOFGYWHQSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "CQYXNTFQGSZODY-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N"
        ]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=False)

        conn = sqlite3.connect(MASSBANK_DB_FN)
        mol_ids_from_query, *_ = zip(
            *conn.execute(
                candidates._get_molecule_feature_by_id_query(molecule_ids, "FCFP__count__all", "fingerprints_meta")
            ).fetchall()
        )
        conn.close()

        # The SQLite query will only return result for unique molecule identifiers (see 'in' operator)
        self.assertEqual(len(set(molecule_ids)), len(mol_ids_from_query))

        # The order in which we provide the molecule ids is preserved in the output. But, if a molecule id appears
        # multiple times, than its first appearance defines its position in the output.
        self.assertEqual(
            (
                "UKQKUSAOCWFKSE-UHFFFAOYSA-N",
                "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
                "TVKGYMYAOVADOP-SOFGYWHQSA-N",
                "HZINSBNIWQLTKI-UHFFFAOYSA-N",
                "LQMQZNHDYCBFJK-UHFFFAOYSA-N",
                "CQYXNTFQGSZODY-UHFFFAOYSA-N"
            ),
            mol_ids_from_query
        )

    def test_get_xlogp3_by_molecule_id(self):
        # -------------------------------
        # InChIKeys with repeated entries
        molecule_ids = [
            "UKQKUSAOCWFKSE-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "TVKGYMYAOVADOP-SOFGYWHQSA-N",
            "HZINSBNIWQLTKI-UHFFFAOYSA-N",
            "UKQKUSAOCWFKSE-UHFFFAOYSA-N",
            "LQMQZNHDYCBFJK-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "TVKGYMYAOVADOP-SOFGYWHQSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N",
            "CQYXNTFQGSZODY-UHFFFAOYSA-N",
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N"
        ]
        xlogp3_ref = [3, 3.4, 3.5, 1.8, 3, 5.1, 3.4, 3.5, 3.4, 3.8, 3.4]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=True)

        # As vector
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False)
        self.assertEqual((len(molecule_ids), ), res.shape)
        np.testing.assert_array_equal(xlogp3_ref, res)

        # As dataframe
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True)
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        self.assertTrue(pd.DataFrame(zip(molecule_ids, xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res))
        self.assertFalse(pd.DataFrame(zip(molecule_ids[::-1], xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res))
        # -------------------------------

        # ------------------------
        # CIDs with missing xlogp3
        molecule_ids = [7389, 66, 77, 5104, 66]
        xlogp3_ref = [np.nan, 3.1, -0.2, np.nan, 3.1]
        xlogp3_ref_imputed = [2.0, 3.1, -0.2, 2.0, 3.1]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="cid", init_with_open_db_conn=True)

        # As vector (raise)
        with self.assertRaises(ValueError):
            candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="raise")

        # As vector (no imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="ignore")
        self.assertEqual((len(molecule_ids), ), res.shape)
        np.testing.assert_array_equal(xlogp3_ref, res)

        # As vector (mean imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="impute_mean")
        self.assertEqual((len(molecule_ids),), res.shape)
        np.testing.assert_array_equal(xlogp3_ref_imputed, res)

        # As dataframe (no imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True, missing_value="ignore")
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        self.assertTrue(pd.DataFrame(zip(molecule_ids, xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res))
        self.assertFalse(
            pd.DataFrame(zip(molecule_ids[::-1], xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res)
        )

        # As dataframe (mean imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True, missing_value="impute_mean")
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        self.assertTrue(
            pd.DataFrame(zip(molecule_ids, xlogp3_ref_imputed), columns=["identifier", "xlogp3"]).equals(res)
        )
        # ------------------------

    def test_get_xlogp3_by_molecule_id__singleton_candidate_set(self):
        # -------------------------------
        # InChIKeys with repeated entries
        molecule_ids = [
            "MIGUTQZXWCSJGD-UHFFFAOYSA-N"
        ]
        xlogp3_ref = [3.4]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=True)

        # As vector
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False)
        self.assertEqual((len(molecule_ids), ), res.shape)
        np.testing.assert_array_equal(xlogp3_ref, res)

        # As dataframe
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True)
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        self.assertTrue(pd.DataFrame(zip(molecule_ids, xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res))
        # -------------------------------

        # ------------------------
        # CIDs with missing xlogp3
        molecule_ids = [77]
        xlogp3_ref = [-0.2]
        xlogp3_ref_imputed = [-0.2]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="cid", init_with_open_db_conn=True)

        # As vector (no imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="ignore")
        self.assertEqual((len(molecule_ids), ), res.shape)
        np.testing.assert_array_equal(xlogp3_ref, res)

        # As vector (mean imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="impute_mean")
        self.assertEqual((len(molecule_ids),), res.shape)
        np.testing.assert_array_equal(xlogp3_ref_imputed, res)

        # As dataframe (no imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True, missing_value="ignore")
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        self.assertTrue(pd.DataFrame(zip(molecule_ids, xlogp3_ref), columns=["identifier", "xlogp3"]).equals(res))

        # As dataframe (mean imputation)
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=True, missing_value="impute_mean")
        self.assertEqual((len(molecule_ids), 2), res.shape)
        self.assertListEqual(["identifier", "xlogp3"], res.columns.tolist())
        # ------------------------

    def test_get_xlogp3_by_molecule_id__all_values_missing(self):
        # Molecule identifiers for which the xlopg3 value is missing
        molecule_ids = [
            "XMVJITFPVVRMHC-UHFFFAOYSA-N",
            "XKNKHVGWJDPIRJ-UHFFFAOYSA-N",
            "FUUFQLXAIUOWML-UHFFFAOYSA-N",
            "FQKUGOMFVDPBIZ-UHFFFAOYSA-N"
        ]
        xlogp3_ref = [np.nan] * len(molecule_ids)

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=True)

        # Ignore the missing values and simply return them
        res = candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="ignore")
        np.testing.assert_array_equal(xlogp3_ref, res)

        # Try to impute the values. Should fail, as all values are missing
        with self.assertRaises(ImputationError):
            candidates.get_xlogp3_by_molecule_id(molecule_ids, return_dataframe=False, missing_value="impute_mean")

    def test_get_molecule_feature_by_molecule_id__repeated_ids(self):
        # ----------
        # InChIKey
        # ----------
        molecule_ids = [
            "AHOUBRCZNHFOSL-UHFFFAOYSA-N",
            "CTNGEIQQUFTGRM-UHFFFAOYSA-N",
            "CTNGEIQQUFTGRM-UHFFFAOYSA-N",
            "AHOUBRCZNHFOSL-UHFFFAOYSA-N",
            "ALMMHVLHNREFEG-UHFFFAOYSA-N",
            "IDJMMERXPGGLMS-ZPFXWGDGSA-N",
            "WYKMDKDFWZAXOH-NVRUNKOVSA-N",
            "ZCZLCPUCJRPPNM-UHFFFAOYSA-N",
            "ZCZLCPUCJRPPNM-UHFFFAOYSA-N",
            "WYKMDKDFWZAXOH-LZZIMNHVSA-N",
            "QPQPNJVRKYVKPW-UHFFFAOYSA-N",
            "IMUZRWAENKOGPB-UHFFFAOYSA-N",
            "WYKMDKDFWZAXOH-LZZIMNHVSA-N",
            "AHOUBRCZNHFOSL-UHFFFAOYSA-N",
            "WYKMDKDFWZAXOH-OWYWQGDKSA-N",
            "WYKMDKDFWZAXOH-LZZIMNHVSA-N",
            "WYKMDKDFWZAXOH-WDRLNXAMSA-N"
        ]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", False)

        np.testing.assert_equal((len(molecule_ids), 1280), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 1280), df_features.iloc[:, 1:].shape)
        self.assertListEqual(molecule_ids, df_features["identifier"].to_list())

        for rep in range(10):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "FCFP__count__all", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # ----------
        # InChIKey1
        # ----------
        molecule_ids = [
            "AHOUBRCZNHFOSL",
            "CTNGEIQQUFTGRM",
            "CTNGEIQQUFTGRM",
            "AHOUBRCZNHFOSL",
            "ALMMHVLHNREFEG",
            "IDJMMERXPGGLMS",
            "WYKMDKDFWZAXOH",
            "ZCZLCPUCJRPPNM",
            "ZCZLCPUCJRPPNM",
            "WYKMDKDFWZAXOH",
            "QPQPNJVRKYVKPW",
            "IMUZRWAENKOGPB",
            "WYKMDKDFWZAXOH",
            "AHOUBRCZNHFOSL",
            "WYKMDKDFWZAXOH",
            "WYKMDKDFWZAXOH",
            "WYKMDKDFWZAXOH"
        ]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", False)

        np.testing.assert_equal((len(molecule_ids), 1280), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 1280), df_features.iloc[:, 1:].shape)
        self.assertListEqual(molecule_ids, df_features["identifier"].to_list())

        for rep in range(10):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "FCFP__count__all", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # --------------------------
        # InChIKey1 (2nd test case)
        # --------------------------
        molecule_ids = [
            "NOSKXGRTKKZRIT",
            "XGYLCEZCVLQHCU",
            "VQZQVLFKPGRLBR",
            "LJPDBPMRVRTPFR",
            "TWSFVQCFVBBJSX",
            "KULXXBYMXPLNPF",
            "VQZQVLFKPGRLBR",
            "GXJJYAMHUBLBSN",
            "IKFXBNOSLPMDFV",
            "KULXXBYMXPLNPF",
            "XGYLCEZCVLQHCU"
        ]

        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "sirius_fps", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "sirius_fps", False)

        np.testing.assert_equal((len(molecule_ids), 3047), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 3047), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(10):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "sirius_fps", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

    def test_get_molecule_feature_by_molecule_id(self):
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")
        molecule_ids = tuple([
            "SBHCLVQMTBWHCD-SPUNCSNDSA-N",
            "LLPVUJKHELUZHR-HLAWJBBLSA-N",
            "DHDMIKWKZFJXOS-UHFFFAOYSA-N",
            "OHVISDOGGOXLEZ-JTQLQIEISA-N",
            "OLUPDFSQBOUUFB-UHFFFAOYSA-N",
            "RLRPRPMUIAMCFN-UHFFFAOYSA-N",
            "UDDHOCQMIMZSLV-HNNXBMFYSA-N",
            "BADLBQAPLMCNBA-UHFFFAOYSA-N"
        ])

        # ---------------------------------
        # BINARY FINGERPRINTS
        # ---------------------------------
        fps_ref = [
            "1,6,7,19,21,23,27,28,30,59,74,107,110,111,112,113,115,187,193,194,195,197,202,207,208,216,219,225,226,228,230,278,282,291,297,344,346,353,359,361,362,368,393,411,416,437,451,459,470,487,503,518,521,523,525,526,529,538,546,639,640,651,669,672,673,674,675,676,677,680,683,692,901,902,904,937,948,997,1000,1043,1070,1078,1083,1088,1090,1138,1195,1298,1318,1338,1478,1509,1559,1672,1841,1884,1885,2033,2108,2138,2357,2407,2788,2834,2850",
            "1,6,12,14,15,19,21,22,24,27,28,29,57,59,101,112,115,149,155,174,184,187,191,192,194,195,197,202,205,206,207,208,216,217,218,219,221,222,223,225,228,230,231,246,249,252,253,258,278,282,283,291,297,299,300,307,310,311,319,320,322,338,344,346,350,353,359,361,362,368,393,399,416,424,429,436,448,451,452,456,459,470,476,485,486,487,498,500,503,504,514,518,521,523,529,530,534,538,539,542,546,547,548,549,554,558,559,560,561,562,563,564,570,571,578,579,586,587,593,594,600,601,614,618,620,632,651,652,672,795,796,808,839,868,876,879,904,917,921,937,948,950,951,952,960,963,966,968,969,971,972,973,974,975,978,1000,1010,1011,1017,1019,1020,1023,1027,1029,1030,1034,1043,1050,1051,1053,1055,1056,1059,1062,1067,1068,1070,1072,1074,1076,1077,1078,1080,1081,1082,1083,1085,1087,1088,1089,1090,1091,1093,1096,1101,1107,1138,1139,1195,1196,1201,1277,1286,1318,1369,1410,1412,1450,1458,1549,1583,1860,1883,1930,2013,2185,2310,2319,2417,2462,2618,2646,2651,2652,2663,2669,2674,2679,2687,2690,2699,2708,2711,2712,2723,2724,2733,2735,2738,2751,2757,2800,2806,2812,2822,2827,2834,2838,2845,2859,2866,2877,2920,2927",
            "1,14,16,19,21,22,27,28,29,37,39,41,59,60,62,110,112,115,119,142,153,155,160,163,167,171,172,173,174,176,180,183,187,189,190,192,196,205,206,207,210,211,215,217,218,221,222,223,224,225,226,227,228,229,230,231,232,246,278,282,283,286,287,288,291,292,293,296,297,299,300,306,307,310,311,319,320,322,328,329,330,338,344,346,350,353,358,361,362,367,375,382,393,399,411,416,418,425,428,429,433,436,446,447,448,451,452,459,464,465,467,470,473,474,476,477,481,482,486,487,488,494,495,497,498,500,503,504,505,507,508,514,515,518,519,521,529,530,532,533,538,539,541,546,548,549,552,553,558,559,560,564,571,583,587,594,604,614,618,619,632,633,651,652,716,717,724,725,757,758,762,771,777,778,782,822,868,869,870,876,884,885,886,904,937,946,960,976,982,986,1000,1018,1019,1027,1043,1066,1067,1068,1070,1099,1104,1111,1126,1127,1131,1132,1133,1135,1138,1160,1178,1179,1187,1190,1195,1206,1208,1209,1233,1245,1249,1250,1255,1394,1410,1603,1634,1697,1712,1839,2145,2192,2228,2255,2262,2566,2618,2653,2663,2664,2676,2704,2711,2723,2751,2758,2782,2812,2866,2877",
            "14,15,19,21,22,27,35,84,87,98,110,112,115,122,148,158,161,168,179,185,186,188,193,195,199,200,204,205,212,213,221,223,224,226,227,228,229,230,231,246,247,248,252,273,278,282,283,287,288,289,291,292,293,296,297,299,300,302,306,307,310,311,312,313,314,315,316,318,322,325,327,328,329,330,332,337,338,344,345,350,353,354,356,360,362,363,365,366,369,375,385,387,390,393,394,397,402,403,411,413,415,416,420,426,428,431,433,436,439,442,448,450,451,452,454,459,461,465,466,467,468,470,471,472,474,477,478,481,482,486,487,489,494,497,498,499,500,503,504,505,514,518,521,522,528,529,530,533,538,539,541,546,548,552,553,558,559,560,565,589,614,618,625,651,749,757,821,868,869,873,931,937,993,1068,1138,1139,1195,1242,1340,1410,1444,1553,1575,1712,1806,2237,2259,2484,2663,2711,2734,2751,2812,2860,2866,2893,2942",
            "1,14,16,19,21,22,27,28,37,59,60,62,98,101,110,112,115,143,153,164,172,173,176,177,178,183,184,187,188,189,190,192,193,194,196,204,205,206,211,212,213,215,217,218,221,222,223,224,225,226,227,228,229,230,231,239,246,252,253,273,278,282,283,287,288,291,293,296,297,299,300,307,310,311,315,316,319,320,322,328,329,330,338,344,346,350,353,356,358,361,362,365,367,374,375,382,393,397,399,402,411,416,429,430,431,433,436,445,448,451,452,459,461,464,465,467,470,471,474,475,476,477,481,482,483,486,487,488,494,497,498,500,501,503,504,505,507,512,514,515,517,518,521,522,529,530,531,533,538,539,540,541,546,548,549,550,551,552,553,558,559,560,565,567,571,574,579,584,589,590,594,598,601,605,614,618,651,652,653,660,724,725,749,757,782,787,817,839,868,869,873,874,904,937,946,982,1000,1027,1034,1043,1068,1077,1111,1138,1160,1188,1195,1202,1206,1227,1233,1241,1256,1325,1339,1344,1441,1486,1587,1646,1660,1661,1860,1924,2026,2262,2306,2329,2512,2555,2566,2618,2646,2650,2653,2655,2669,2674,2676,2690,2691,2704,2724,2733,2734,2750,2758,2782,2787,2796,2849,2857,2866,2877,2893,2937",
            "5,6,7,9,14,15,19,21,22,27,28,29,30,34,56,102,107,112,115,129,138,149,155,157,170,172,181,183,184,191,205,206,207,211,216,217,221,223,228,230,231,246,252,253,258,259,278,282,283,285,291,297,338,344,350,353,362,382,393,416,451,459,467,470,482,485,487,500,502,504,518,521,529,530,538,539,540,546,547,548,549,550,551,558,559,560,593,594,597,614,651,652,795,797,868,901,906,921,923,937,956,960,970,975,982,997,1003,1029,1043,1055,1056,1061,1070,1078,1083,1088,1090,1093,1096,1101,1107,1138,1139,1195,1333,1338,1407,1495,1597,1769,1785,2300,2383,2515,2651,2666,2682,2697,2809,2834,2838,2839,2852,2928,2944",
            "14,16,19,21,22,28,29,37,39,42,87,98,101,112,115,145,148,156,164,166,167,168,172,174,177,180,184,186,190,192,199,200,201,204,205,206,207,208,215,219,220,223,224,226,227,228,229,230,231,236,237,240,241,242,246,252,253,273,275,278,282,283,288,291,292,296,297,299,300,301,302,305,306,307,310,311,312,313,315,316,319,320,322,323,324,325,328,329,330,332,335,336,337,338,344,345,350,351,353,354,356,360,362,363,365,369,375,382,385,393,394,397,399,402,411,413,415,416,417,418,420,426,427,428,429,431,433,436,439,448,451,452,454,455,459,461,464,465,467,470,471,474,476,477,481,486,487,494,497,498,500,503,504,505,514,515,518,521,522,529,530,532,533,538,539,541,546,547,548,549,552,558,559,560,562,571,581,591,594,603,614,651,652,738,743,822,838,855,856,868,876,909,912,924,937,1005,1043,1063,1064,1070,1075,1099,1104,1105,1111,1112,1113,1126,1132,1160,1173,1178,1195,1206,1250,1361,1417,1484,1534,1547,1720,1761,1767,1803,1807,1910,2032,2118,2202,2228,2229,2240,2280,2463,2591,2597,2613,2647,2683,2702,2724,2750,2810,2812,2866,2877,2880,2893,2917,2947",
            "6,7,14,16,19,21,22,27,28,34,37,101,112,115,163,171,172,183,184,187,188,192,193,194,195,197,204,205,206,207,208,211,215,216,219,223,224,226,228,230,231,246,252,253,273,278,282,283,291,293,297,299,300,307,310,311,319,320,322,323,338,344,350,353,362,382,393,399,411,416,429,436,448,449,451,452,459,464,467,470,474,476,482,485,486,487,488,498,500,503,504,514,518,521,529,530,531,538,539,546,548,549,558,559,560,562,564,571,578,587,594,600,651,672,673,674,675,677,680,683,691,692,698,711,723,749,822,868,919,937,1043,1070,1078,1083,1088,1093,1096,1101,1107,1138,1195,1206,1509,1544,1653,1670,1744,1763,1803,1871,2240,2329,2407,2653,2750,2866,2877"
        ]
        fps_mat_ref = np.zeros((len(fps_ref), 3047))
        for i, fp_i in enumerate(fps_ref):
            for j in fp_i.split(","):
                fps_mat_ref[i, int(j)] = 1

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "sirius_fps", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "sirius_fps", False)

        np.testing.assert_equal(fps_mat_ref, feature_matrix)
        np.testing.assert_equal(feature_matrix, df_features.iloc[:, 1:].values)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(10):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Feature matrix
            feature_matrix_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "sirius_fps")
            np.testing.assert_array_equal(feature_matrix[rnd_idc], feature_matrix_shf)

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "sirius_fps", return_dataframe=True)
            np.testing.assert_equal(feature_matrix[rnd_idc], df_features_shf.iloc[:, 1:].values)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # ---------------------------------
        # COUNTING FINGERPRINTS
        # ---------------------------------
        fps_ref_bits = [
            "0,1,2,24,35,38,50,51,53,150,221,223,293,373,376,574,785,961",
            "0,2,3,4,9,13,17,18,22,24,51,64,71,84,96,97,103,150,158,159,162,167,199,233,270,469,471,920,1114,1192",
            "0,1,2,4,9,18,21,22,23,24,34,51,61,68,71,77,78,79,80,83,85,106,107,110,111,112,115,126,139,287,297,318,380,476,480",
            "0,2,4,9,21,22,32,33,34,40,42,51,52,56,76,90,95,96,196,200,316,387,527,550,554,762",
            "0,1,2,4,9,16,18,21,22,51,68,79,85,86,88,89,94,95,107,114,115,167,251,281,287,299,380,504,569,582,588,643,696,819",
            "0,2,3,19,24,51,96,113,116,126,180,204,375,464,640,641,781,928,1037,1203,1228",
            "0,2,4,5,6,9,10,21,22,24,27,33,40,47,52,60,68,71,79,80,88,104,115,237,238,239,242,249,267,328,349,423,757,798,896,974,975,976,977,978",
            "0,2,3,4,19,22,24,25,51,68,79,95,104,114,115,150,267,272,293,383,384,653,790,798,799"
        ]
        fps_ref_vals = [
            "19,1,1,8,1,1,1,1,1,6,5,1,1,1,1,4,1,3",
            "15,2,1,6,3,1,1,1,2,4,4,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "12,1,3,6,1,1,1,3,1,2,1,3,1,2,1,2,1,2,1,1,1,1,1,1,2,1,1,1,1,1,1,1,1,1,1",
            "4,1,10,1,2,4,2,1,1,1,1,2,1,2,2,1,1,1,1,1,2,1,2,1,1,2",
            "8,1,4,12,2,1,1,1,6,2,3,3,1,1,2,1,1,2,1,2,1,1,1,2,1,1,1,1,1,1,1,1,1,1",
            "20,1,1,1,6,4,1,1,1,2,1,3,1,1,1,2,2,1,1,2,1",
            "6,1,14,1,1,2,1,1,7,1,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1",
            "11,2,1,10,1,6,7,1,1,2,2,2,2,2,2,3,2,2,1,2,2,1,1,2,2"
        ]

        fps_mat_ref = np.zeros((len(fps_ref), 1280))
        for i in range(len(fps_ref)):
            for bit, val in zip(fps_ref_bits[i].split(","), fps_ref_vals[i].split(",")):
                fps_mat_ref[i, int(bit)] = int(val)

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "FCFP__count__all", False)

        np.testing.assert_equal(fps_mat_ref, feature_matrix)
        np.testing.assert_equal(feature_matrix, df_features.iloc[:, 1:].values)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(10):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Feature matrix
            feature_matrix_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "FCFP__count__all")
            np.testing.assert_array_equal(feature_matrix[rnd_idc], feature_matrix_shf)

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "FCFP__count__all", return_dataframe=True)
            np.testing.assert_equal(feature_matrix[rnd_idc], df_features_shf.iloc[:, 1:].values)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

    def test_molecule_feature_transformation(self):
        # Load Bouwmeester molecular descriptors for all candidates of a certain spectrum and train a transformer
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952"})
        candidates = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey1")

        desc = candidates.get_molecule_features(spectrum, features="bouwmeester__smiles_can")
        self.assertEqual((1923, 196), desc.shape)
        self.assertFalse(np.any(np.isnan(desc)))

        # Train a variance filter
        var_threshold = 0.01 ** 2
        var_trans = VarianceThreshold(threshold=var_threshold).fit(desc)

        for idx, ft in enumerate([var_trans, {"bouwmeester__smiles_can": var_trans}, None, {"bla": var_trans}]):
            candidates_w_trans = CandSQLiteDB_Massbank(
                MASSBANK_DB_FN, molecule_identifier="inchikey1", feature_transformer=ft
            )
            desc_trans = candidates_w_trans.get_molecule_features(spectrum, features="bouwmeester__smiles_can")

            if idx in [0, 1]:
                # Transformation is applied
                self.assertEqual((1923, np.sum(var_trans.get_support())), desc_trans.shape)
                self.assertTrue(np.all(np.var(desc_trans, axis=0) > var_threshold))
            else:
                # Transformation is not applied
                self.assertEqual((1923, 196), desc.shape)
                self.assertEqual(0, np.min(np.var(desc_trans, axis=0)))

        # Train a filter pipeline
        pipeline = Pipeline([
            ("var_filter", VarianceThreshold(threshold=var_threshold)),
            ("std_scaler", StandardScaler())
        ]).fit(desc)

        for idx, ft in enumerate([pipeline, {"bouwmeester__smiles_can": pipeline}, None, {"bla": pipeline}]):
            candidates_w_trans = CandSQLiteDB_Massbank(
                MASSBANK_DB_FN, molecule_identifier="inchikey1", feature_transformer=ft
            )
            desc_trans = candidates_w_trans.get_molecule_features(spectrum, features="bouwmeester__smiles_can")

            if idx in [0, 1]:
                # Transformation is applied
                n_feat = np.sum(pipeline.named_steps["var_filter"].get_support())
                self.assertEqual((1923, n_feat), desc_trans.shape)
                self.assertTrue(np.all(np.var(desc_trans, axis=0) > var_threshold))
                self.assertTrue(np.all(np.isclose(np.mean(desc_trans, axis=0), 0)))
            else:
                # Transformation is not applied
                self.assertEqual((1923, 196), desc.shape)
                self.assertEqual(0, np.min(np.var(desc_trans, axis=0)))
                np.testing.assert_array_equal(np.mean(desc, axis=0), np.mean(desc_trans, axis=0))

    def test_ensure_feature_is_available(self):
        candidates = CandSQLiteDB_Massbank(db_fn=MASSBANK_DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_feature_is_available("bla")
            candidates._ensure_feature_is_available(None)
            candidates._ensure_feature_is_available("")
            candidates._ensure_feature_is_available("substructure_count")

        self.assertEqual("fingerprints_meta", candidates._ensure_feature_is_available("FCFP__count__all"))
        self.assertEqual("fingerprints_meta", candidates._ensure_feature_is_available("sirius_fps"))
        self.assertEqual("descriptors_meta", candidates._ensure_feature_is_available("bouwmeester__smiles_can"))

    def test_ensure_molecule_identifier_is_available(self):
        candidates = CandSQLiteDB_Massbank(db_fn=MASSBANK_DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_molecule_identifier_is_available("inchistr")
            candidates._ensure_molecule_identifier_is_available(None)
            candidates._ensure_molecule_identifier_is_available("")

        candidates._ensure_molecule_identifier_is_available("inchi")
        candidates._ensure_molecule_identifier_is_available("inchikey")
        candidates._ensure_molecule_identifier_is_available("cid")

    def test_ensure_ms2_scorer_is_available(self):
        candidates = CandSQLiteDB_Massbank(db_fn=MASSBANK_DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_ms2scorer_is_available("super_metfrag")
            candidates._ensure_ms2scorer_is_available(None)
            candidates._ensure_ms2scorer_is_available("")

        candidates._ensure_ms2scorer_is_available("sirius__sd__correct_mf")
        candidates._ensure_ms2scorer_is_available("metfrag__norm_after_merge")


class TestCandidateSQLiteDB(unittest.TestCase):
    @staticmethod
    @delayed
    def _get_molecule_features_by_molecule_id(candidates, molecule_ids, features):
        with candidates:
            feature_mat = candidates.get_molecule_features_by_molecule_id(molecule_ids, features)

        return feature_mat

    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})

        # Molecule identifier: inchikey
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(2233, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1918, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

    def test_get_ms2_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is IOKR
        scores = candidates.get_ms_scores(spectrum, ms_scorer="IOKR__696a17f3")
        self.assertEqual(2233, len(scores))
        self.assertEqual(0.00006426196626211542, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(2233, len(scores))
        np.testing.assert_allclose(0.000028105936908001407, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is IOKR
        scores = candidates.get_ms_scores(spectrum, ms_scorer="IOKR__696a17f3")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.7441697188705315, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms_scores(spectrum, ms_scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.18165470462229025, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

    def test_get_molecule_features(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-019"})
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey1")

        # IOKR features
        fps = candidates.get_molecule_features(spectrum, features="iokr_fps__positive")
        self.assertEqual((5103, 7936), fps.shape)
        self.assertTrue(np.all(np.isin(fps, [0, 1])))

        # Substructure features
        fps = candidates.get_molecule_features(spectrum, features="substructure_count")
        self.assertEqual((5103, 307), fps.shape)
        self.assertTrue(np.all(fps >= 0))

    def test_get_molecule_feature_by_id_query(self):
        molecule_ids = [
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ]

        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=False)

        conn = sqlite3.connect(BACH2020_DB_FN)
        mol_ids_from_query, _ = zip(
            *conn.execute(
                candidates._get_molecule_feature_by_id_query(molecule_ids, "substructure_count", "fingerprints_meta")
            ).fetchall()
        )
        conn.close()

        # The SQLite query will only return result for unique molecule identifiers (see 'in' operator)
        self.assertEqual(len(set(molecule_ids)), len(mol_ids_from_query))

        # The order in which we provide the molecule ids is preserved in the output. But, if a molecule id appears
        # multiple times, than its first appearance defines its position in the output.
        self.assertEqual(
            (
                "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
                "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
                "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
                "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
                "BFFTVFNQHIJUQT-INIZCTEOSA-N",
                "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
                "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
                "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
                "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
            ),
            mol_ids_from_query
        )

    def test_get_molecule_feature_by_molecule_id__repeated_ids(self):
        # ----------
        # InChIKey
        # ----------
        molecule_ids = [
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ]

        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", False)

        np.testing.assert_equal((len(molecule_ids), 7936), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 7936), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "iokr_fps__positive", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # ----------
        # InChIKey1
        # ----------
        molecule_ids = [
            "CEJQCDWBMMEZFJ",
            "FGXWKSZFVQUSTL",
            "BFFTVFNQHIJUQT",
            "ATFVEXNQLNTOHV",
            "BFFTVFNQHIJUQT",
            "CEJQCDWBMMEZFJ",
            "BFFTVFNQHIJUQT",
            "ATFVEXNQLNTOHV",
            "BFFTVFNQHIJUQT",
            "CEJQCDWBMMEZFJ",
            "AZVUYIUQQIFCQK",
            "AZVUYIUQQIFCQK"
        ]

        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey1")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", False)

        np.testing.assert_equal((len(molecule_ids), 7936), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 7936), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "iokr_fps__positive", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # --------------------------
        # InChIKey1 (2nd test case)
        # --------------------------
        molecule_ids = [
            "CEJQCDWBMMEZFJ",
            "FGXWKSZFVQUSTL",
            "BFFTVFNQHIJUQT",
            "ATFVEXNQLNTOHV",
            "BFFTVFNQHIJUQT",
            "CEJQCDWBMMEZFJ",
            "BFFTVFNQHIJUQT",
            "ATFVEXNQLNTOHV",
            "BFFTVFNQHIJUQT",
            "CEJQCDWBMMEZFJ",
            "AZVUYIUQQIFCQK",
            "AZVUYIUQQIFCQK"
        ]

        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey1")
        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "substructure_count", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "substructure_count", False)

        np.testing.assert_equal((len(molecule_ids), 307), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 307), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "substructure_count", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

    def test_get_molecule_feature_by_molecule_id__ids_as_tuple(self):
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")

        # ---------------------------------
        # REPEATED IDS
        # ---------------------------------
        molecule_ids = tuple([
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ])

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", False)

        np.testing.assert_equal((len(molecule_ids), 7936), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 7936), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "iokr_fps__positive", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # ---------------------------------
        # UNIQUE IDS
        # ---------------------------------
        molecule_ids = tuple([
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ])

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", False)

        np.testing.assert_equal((len(molecule_ids), 7936), feature_matrix.shape)
        np.testing.assert_equal((len(molecule_ids), 7936), df_features.iloc[:, 1:].shape)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple([molecule_ids[i] for i in rnd_idc]), "iokr_fps__positive", return_dataframe=True)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

    def test_get_molecule_feature_by_molecule_id(self):
        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")
        molecule_ids = tuple([
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ])

        # ---------------------------------
        # BINARY FINGERPRINTS
        # ---------------------------------
        fps_ref = [
            "26,33,44,45,47,48,65,86,89,212,232,234,243,244,247,337,338,342,358,363,364,365,370,420,445,448,458,460,462,463,466,468,469,470,473,474,479,481,483,486,488,489,490,493,494,496,500,501,503,504,505,508,510,511,512,514,516,517,520,521,522,525,526,527,528,530,531,533,536,538,539,540,541,542,544,545,546,547,548,576,577,578,585,586,587,588,590,591,592,594,595,613,719,721,722,726,728,729,754,755,756,757,761,762,768,831,832,833,834,835,837,859,860,861,862,870,884,908,914,916,918,920,921,927,931,933,934,938,941,946,947,951,952,953,955,960,962,963,965,966,967,972,973,979,982,992,994,1007,1010,1013,1017,1018,1022,1025,1026,1029,1040,1046,1048,1058,1060,1061,1063,1066,1071,1075,1077,1078,1079,1080,1082,1092,1095,1096,1097,1099,1100,1106,1114,1115,1116,1121,1122,1123,1125,1126,1128,1131,1132,1133,1136,1140,1145,1146,1148,1152,1154,1158,1160,1161,1167,1168,1169,1171,1172,1174,1175,1176,1179,1183,1184,1187,1189,1194,1204,1209,1210,1212,1216,1230,1232,1233,1236,1240,1241,1244,1250,1253,1254,1255,1259,1305,1326,1346,1368,1389,1409,1472,1703,1768,1769,1812,1815,1909,2076,2091,2109,2619,3037,3116,3258,3662,3730,3733,4420,4446,4628,4695,5111,5163,5177,5200,5204,5211,5212,5221,5222,5223,5228,5244,5253,5292,5340,5353,5355,5386,5397,5551,5802,6392,6404,6435,6467,6479,6594,6697,6702,6766,6787,6898,6901,6919,6929,6934,6985,7004,7011,7108,7116,7138,7150,7229,7235,7243,7294,7371,7386,7404,7550,7681,7716,7730,7786,7820,7836",
            "2,26,27,36,37,44,45,47,48,50,64,65,148,151,206,274,277,337,338,350,358,363,364,365,370,415,416,420,434,436,438,441,442,443,444,447,450,455,456,458,460,461,464,471,472,473,474,475,477,478,480,481,483,485,487,489,492,493,494,495,497,498,500,501,502,504,505,506,507,509,511,512,513,514,515,519,520,521,522,523,525,528,529,530,531,532,533,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,754,755,756,757,761,831,859,860,861,862,869,881,884,908,909,914,916,920,921,922,927,928,929,931,932,941,942,946,947,950,951,956,959,960,966,981,982,990,992,994,996,1006,1010,1016,1017,1018,1019,1022,1026,1028,1029,1033,1035,1037,1041,1046,1059,1063,1066,1092,1096,1100,1107,1108,1111,1112,1115,1116,1128,1132,1140,1142,1143,1146,1154,1155,1158,1160,1168,1171,1175,1178,1179,1184,1189,1190,1191,1194,1210,1213,1216,1232,1235,1236,1238,1239,1240,1244,1253,1254,1255,1256,1259,1260,1264,1265,1268,1284,1285,1286,1291,1354,1472,1491,1689,1768,1769,1773,1812,1815,1817,1853,1937,2118,2125,2140,2619,2664,3113,3195,3221,3733,4018,4144,4252,4304,4326,4357,4420,4446,4484,4496,4628,4695,4799,4873,4878,4885,4891,4914,4926,5013,5014,5111,5125,5126,5132,5138,5142,5163,5168,5172,5177,5179,5180,5183,5187,5189,5198,5200,5201,5204,5207,5208,5211,5214,5215,5221,5237,5239,5244,5253,5256,5257,5258,5259,5260,5261,5266,5271,5277,5353,5372,5378,5397,5408,5409,5419,5426,5427,5428,5430,5437,5442,5551,5754,5757,5766,5791,5797,5799,5800,5802,6062,6064,6141,6151,6166,6299,6300,6301,6314,6324,6340,6404,6406,6491,6516,6531,6535,6547,6607,6661,6702,6775,6841,6935,7001,7004,7011,7057,7068,7140,7159,7229,7235,7300,7331,7365,7404,7444,7458,7460,7491,7681,7730,7754,7826,7836,7860,7924",
            "2,26,27,36,37,44,45,47,48,50,64,65,67,148,151,206,212,274,277,337,338,350,358,363,365,368,370,415,416,420,434,438,441,442,443,444,447,449,450,452,455,456,457,458,460,461,462,464,465,466,468,471,472,473,474,475,477,478,479,480,483,485,487,489,491,493,494,495,500,501,502,503,504,505,506,507,509,510,513,514,515,519,520,521,522,523,524,525,526,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,719,724,725,754,755,831,859,860,861,862,869,875,881,884,908,909,910,915,917,920,921,922,927,928,929,931,932,941,942,946,947,950,951,956,959,960,966,967,969,981,982,990,992,994,996,1006,1010,1015,1016,1017,1019,1022,1026,1027,1028,1029,1033,1035,1038,1041,1046,1059,1063,1066,1092,1095,1096,1100,1104,1107,1108,1111,1113,1116,1128,1132,1140,1142,1145,1146,1154,1156,1158,1160,1171,1172,1175,1179,1181,1184,1187,1189,1190,1191,1194,1210,1214,1216,1219,1221,1222,1232,1235,1236,1240,1244,1253,1254,1255,1264,1272,1273,1284,1285,1286,1287,1288,1291,1354,1472,1485,1491,1543,1569,1768,1812,1854,1867,1872,1881,1938,1963,1975,2028,2081,2140,2148,2249,2610,2615,2616,2619,2625,2627,2631,2644,2664,3113,3143,3144,3195,3733,3736,4018,4019,4144,4252,4302,4326,4357,4420,4446,4484,4488,4496,4695,4766,4799,4802,4815,4820,4840,4853,4879,4881,4905,4906,4926,5013,5014,5111,5118,5119,5125,5126,5127,5142,5153,5211,5212,5214,5215,5221,5222,5223,5239,5244,5246,5248,5249,5252,5253,5257,5258,5259,5260,5261,5266,5277,5353,5355,5359,5360,5370,5378,5397,5399,5403,5413,5417,5419,5426,5427,5428,5430,5442,5551,5662,5708,5761,5762,5764,5765,5766,5791,5797,5799,5800,5802,6004,6062,6146,6166,6282,6314,6324,6340,6404,6483,6491,6537,6547,6607,6661,6666,6702,6754,6824,6825,6919,7004,7068,7121,7147,7235,7273,7300,7326,7343,7404,7491,7550,7632,7637,7681,7687,7727,7730,7754,7836",
            "2,26,27,36,37,44,45,47,48,50,64,65,148,151,206,274,277,337,338,350,358,363,364,365,370,415,416,420,434,436,438,441,442,443,444,447,450,455,456,458,460,461,464,471,472,473,474,475,477,478,480,481,483,485,487,489,492,493,494,495,497,498,500,501,502,504,505,506,507,509,511,512,513,514,515,519,520,521,522,523,525,528,529,530,531,532,533,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,754,755,756,757,761,831,859,860,861,862,869,881,884,908,909,914,916,920,921,922,927,928,929,931,932,941,942,946,947,950,951,956,959,960,966,981,982,990,992,994,996,1006,1010,1016,1017,1018,1019,1022,1026,1028,1029,1033,1035,1037,1041,1046,1059,1063,1066,1092,1096,1100,1107,1108,1111,1112,1115,1116,1128,1132,1140,1142,1143,1146,1154,1155,1158,1160,1168,1171,1175,1178,1179,1184,1189,1190,1191,1194,1210,1213,1216,1232,1235,1236,1238,1239,1240,1244,1253,1254,1255,1256,1259,1260,1264,1265,1268,1284,1285,1286,1291,1354,1472,1491,1689,1768,1769,1773,1812,1815,1817,1853,1937,2118,2125,2140,2619,2664,3113,3195,3221,3733,4018,4144,4252,4304,4326,4357,4420,4446,4484,4496,4628,4695,4799,4873,4878,4885,4891,4914,4926,5013,5014,5111,5125,5126,5132,5138,5142,5163,5168,5172,5177,5179,5180,5183,5187,5189,5198,5200,5201,5204,5207,5208,5211,5214,5215,5221,5237,5239,5244,5253,5256,5257,5258,5259,5260,5261,5266,5271,5277,5353,5372,5378,5397,5408,5409,5419,5426,5427,5428,5430,5437,5442,5551,5754,5757,5766,5791,5797,5799,5800,5802,6062,6064,6141,6151,6166,6299,6300,6301,6314,6324,6340,6404,6406,6491,6516,6531,6535,6547,6607,6661,6702,6775,6841,6935,7001,7004,7011,7057,7068,7140,7159,7229,7235,7300,7331,7365,7404,7444,7458,7460,7491,7681,7730,7754,7826,7836,7860,7924",
            "2,4,36,38,44,45,48,64,68,81,112,200,244,245,247,337,338,342,350,358,365,366,370,417,433,440,448,459,466,472,475,478,479,480,482,488,492,493,496,497,500,503,504,508,509,510,520,521,523,526,528,529,535,536,537,538,539,540,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,594,595,596,719,721,722,754,755,761,762,768,769,831,832,833,835,837,859,860,861,862,908,909,916,917,920,922,927,928,931,932,933,934,942,946,947,950,952,953,957,958,960,962,963,966,972,979,981,985,992,994,996,1006,1007,1008,1010,1013,1015,1017,1018,1019,1022,1025,1027,1029,1037,1040,1046,1048,1052,1058,1066,1068,1069,1070,1071,1073,1074,1078,1082,1092,1096,1099,1100,1106,1111,1112,1114,1115,1117,1118,1119,1121,1122,1124,1125,1128,1129,1131,1132,1140,1141,1142,1146,1149,1150,1152,1154,1155,1156,1157,1158,1160,1161,1164,1165,1168,1170,1171,1173,1175,1176,1178,1179,1180,1182,1183,1184,1188,1189,1190,1191,1194,1195,1196,1199,1202,1204,1208,1209,1210,1213,1216,1217,1218,1221,1227,1231,1232,1233,1235,1236,1240,1241,1242,1244,1247,1248,1253,1254,1255,1256,1259,1260,1264,1265,1268,1269,1272,1274,1280,1281,1284,1285,1286,1290,1339,1353,1402,1609,1768,1810,1937,2118,2140,2664,2923,3037,3113,3664,4018,4047,4066,4182,4192,4446,4457,4481,4496,4666,4678,4695,4752,4754,4911,4926,5031,5152,5153,5259,5427,5471,5802,5806,6188,6281,6387,6406,6457,6467,6468,6491,6540,6622,6641,6661,6697,6702,6708,6787,6816,6919,6934,6964,7035,7068,7138,7235,7300,7304,7371,7404,7436,7491,7530,7585,7758,7810,7835,7836,7842",
            "2,26,27,36,37,44,45,47,48,50,64,65,148,151,206,274,277,337,338,350,358,363,364,365,370,415,416,420,434,436,438,441,442,443,444,447,450,455,456,458,460,461,464,471,472,473,474,475,477,478,480,481,483,485,487,489,492,493,494,495,497,498,500,501,502,504,505,506,507,509,511,512,513,514,515,519,520,521,522,523,525,528,529,530,531,532,533,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,754,755,756,757,761,831,859,860,861,862,869,881,884,908,909,914,916,920,921,922,927,928,929,931,932,941,942,946,947,950,951,956,959,960,966,981,982,990,992,994,996,1006,1010,1016,1017,1018,1019,1022,1026,1028,1029,1033,1035,1037,1041,1046,1059,1063,1066,1092,1096,1100,1107,1108,1111,1112,1115,1116,1128,1132,1140,1142,1143,1146,1154,1155,1158,1160,1168,1171,1175,1178,1179,1184,1189,1190,1191,1194,1210,1213,1216,1232,1235,1236,1238,1239,1240,1244,1253,1254,1255,1256,1259,1260,1264,1265,1268,1284,1285,1286,1291,1354,1472,1491,1689,1768,1769,1773,1812,1815,1817,1853,1937,2118,2125,2140,2619,2664,3113,3195,3221,3733,4018,4144,4252,4304,4326,4357,4420,4446,4484,4496,4628,4695,4799,4873,4878,4885,4891,4914,4926,5013,5014,5111,5125,5126,5132,5138,5142,5163,5168,5172,5177,5179,5180,5183,5187,5189,5198,5200,5201,5204,5207,5208,5211,5214,5215,5221,5237,5239,5244,5253,5256,5257,5258,5259,5260,5261,5266,5271,5277,5353,5372,5378,5397,5408,5409,5419,5426,5427,5428,5430,5437,5442,5551,5754,5757,5766,5791,5797,5799,5800,5802,6062,6064,6141,6151,6166,6299,6300,6301,6314,6324,6340,6404,6406,6491,6516,6531,6535,6547,6607,6661,6702,6775,6841,6935,7001,7004,7011,7057,7068,7140,7159,7229,7235,7300,7331,7365,7404,7444,7458,7460,7491,7681,7730,7754,7826,7836,7860,7924",
            "2,26,27,36,37,44,45,47,48,50,64,65,67,148,151,206,212,274,277,337,338,350,358,363,365,368,370,415,416,420,434,438,441,442,443,444,447,449,450,452,455,456,457,458,460,461,462,464,465,466,468,471,472,473,474,475,477,478,479,480,483,485,487,489,491,493,494,495,500,501,502,503,504,505,506,507,509,510,513,514,515,519,520,521,522,523,524,525,526,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,719,724,725,754,755,831,859,860,861,862,869,875,881,884,908,909,910,915,917,920,921,922,927,928,929,931,932,941,942,946,947,950,951,956,959,960,966,967,969,981,982,990,992,994,996,1006,1010,1015,1016,1017,1019,1022,1026,1027,1028,1029,1033,1035,1038,1041,1046,1059,1063,1066,1092,1095,1096,1100,1104,1107,1108,1111,1113,1116,1128,1132,1140,1142,1145,1146,1154,1156,1158,1160,1171,1172,1175,1179,1181,1184,1187,1189,1190,1191,1194,1210,1214,1216,1219,1221,1222,1232,1235,1236,1240,1244,1253,1254,1255,1264,1272,1273,1284,1285,1286,1287,1288,1291,1354,1472,1485,1491,1543,1569,1768,1812,1854,1867,1872,1881,1938,1963,1975,2028,2081,2140,2148,2249,2610,2615,2616,2619,2625,2627,2631,2644,2664,3113,3143,3144,3195,3733,3736,4018,4019,4144,4252,4302,4326,4357,4420,4446,4484,4488,4496,4695,4766,4799,4802,4815,4820,4840,4853,4879,4881,4905,4906,4926,5013,5014,5111,5118,5119,5125,5126,5127,5142,5153,5211,5212,5214,5215,5221,5222,5223,5239,5244,5246,5248,5249,5252,5253,5257,5258,5259,5260,5261,5266,5277,5353,5355,5359,5360,5370,5378,5397,5399,5403,5413,5417,5419,5426,5427,5428,5430,5442,5551,5662,5708,5761,5762,5764,5765,5766,5791,5797,5799,5800,5802,6004,6062,6146,6166,6282,6314,6324,6340,6404,6483,6491,6537,6547,6607,6661,6666,6702,6754,6824,6825,6919,7004,7068,7121,7147,7235,7273,7300,7326,7343,7404,7491,7550,7632,7637,7681,7687,7727,7730,7754,7836",
            "2,4,36,38,44,45,48,64,68,81,112,200,244,245,247,337,338,342,350,358,365,366,370,417,433,440,448,459,466,472,475,478,479,480,482,488,492,493,496,497,500,503,504,508,509,510,520,521,523,526,528,529,535,536,537,538,539,540,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,594,595,596,719,721,722,754,755,761,762,768,769,831,832,833,835,837,859,860,861,862,908,909,916,917,920,922,927,928,931,932,933,934,942,946,947,950,952,953,957,958,960,962,963,966,972,979,981,985,992,994,996,1006,1007,1008,1010,1013,1015,1017,1018,1019,1022,1025,1027,1029,1037,1040,1046,1048,1052,1058,1066,1068,1069,1070,1071,1073,1074,1078,1082,1092,1096,1099,1100,1106,1111,1112,1114,1115,1117,1118,1119,1121,1122,1124,1125,1128,1129,1131,1132,1140,1141,1142,1146,1149,1150,1152,1154,1155,1156,1157,1158,1160,1161,1164,1165,1168,1170,1171,1173,1175,1176,1178,1179,1180,1182,1183,1184,1188,1189,1190,1191,1194,1195,1196,1199,1202,1204,1208,1209,1210,1213,1216,1217,1218,1221,1227,1231,1232,1233,1235,1236,1240,1241,1242,1244,1247,1248,1253,1254,1255,1256,1259,1260,1264,1265,1268,1269,1272,1274,1280,1281,1284,1285,1286,1290,1339,1353,1402,1609,1768,1810,1937,2118,2140,2664,2923,3037,3113,3664,4018,4047,4066,4182,4192,4446,4457,4481,4496,4666,4678,4695,4752,4754,4911,4926,5031,5152,5153,5259,5427,5471,5802,5806,6188,6281,6387,6406,6457,6467,6468,6491,6540,6622,6641,6661,6697,6702,6708,6787,6816,6919,6934,6964,7035,7068,7138,7235,7300,7304,7371,7404,7436,7491,7530,7585,7758,7810,7835,7836,7842",
            "2,36,38,44,45,47,48,65,66,81,151,161,163,190,274,277,337,338,350,358,363,365,368,370,405,415,416,434,437,438,441,443,444,447,450,455,456,458,462,464,465,469,471,473,474,475,476,477,478,480,481,483,485,489,493,494,495,496,500,501,504,505,507,509,510,511,512,513,514,516,518,519,520,521,523,525,526,527,528,529,530,531,532,533,534,535,536,537,538,539,540,541,542,543,544,545,546,547,548,576,577,578,585,586,587,588,590,591,594,595,596,609,691,692,754,755,756,757,761,831,859,860,861,862,869,875,881,908,909,911,916,917,920,921,925,927,928,929,931,941,942,944,946,947,950,952,953,957,958,960,966,968,969,981,990,992,996,1010,1013,1015,1017,1019,1022,1025,1027,1033,1035,1040,1041,1046,1052,1066,1071,1074,1078,1092,1096,1100,1104,1112,1116,1117,1118,1120,1121,1122,1124,1125,1128,1132,1140,1141,1143,1145,1146,1149,1150,1154,1158,1160,1161,1165,1168,1170,1171,1175,1176,1178,1179,1180,1182,1183,1184,1187,1189,1194,1195,1202,1204,1206,1209,1210,1213,1214,1216,1217,1219,1221,1222,1227,1228,1231,1232,1233,1236,1238,1240,1241,1242,1244,1253,1254,1255,1256,1258,1259,1265,1267,1278,1279,1284,1286,1297,1318,1337,1360,1381,1400,1472,1544,1699,1703,1768,1769,1812,1938,1951,2076,2094,2140,2148,2229,2310,2664,2889,3116,3287,3733,4018,4138,4183,4185,4252,4326,4327,4446,4496,4529,4628,4695,4766,4840,4867,4869,4926,5111,5130,5131,5163,5177,5200,5204,5206,5207,5211,5213,5214,5221,5237,5244,5246,5252,5253,5254,5256,5258,5280,5285,5286,5353,5372,5397,5399,5413,5414,5416,5426,5427,5471,5476,5477,5490,5492,5494,5551,5708,5725,5731,5752,5757,5765,5791,5799,5800,5802,5943,6064,6206,6222,6241,6314,6404,6428,6483,6491,6497,6547,6607,6661,6702,6797,6948,7011,7068,7154,7162,7222,7229,7235,7243,7255,7282,7300,7379,7396,7404,7491,7616,7681,7730,7836,7914"
        ]
        fps_ref = [fp.split(",") for fp in fps_ref]

        fps_mat_ref = np.zeros((len(fps_ref), 7936))
        for i in range(len(fps_ref)):
            for j in range(len(fps_ref[i])):
                fps_mat_ref[i, int(fps_ref[i][j])] = 1

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "iokr_fps__positive", False)

        np.testing.assert_equal(fps_mat_ref, feature_matrix)
        np.testing.assert_equal(feature_matrix, df_features.iloc[:, 1:].values)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Feature matrix
            feature_matrix_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "iokr_fps__positive")
            np.testing.assert_array_equal(feature_matrix[rnd_idc], feature_matrix_shf)

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "iokr_fps__positive", return_dataframe=True)
            np.testing.assert_equal(feature_matrix[rnd_idc], df_features_shf.iloc[:, 1:].values)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

        # ---------------------------------
        # COUNTING FINGERPRINTS
        # ---------------------------------
        fps_ref = [
            "1: 3, 22: 1, 25: 1, 148: 2, 168: 2, 170: 1, 179: 2, 180: 2, 183: 4, 273: 18, 274: 5, 278: 2, 294: 15, 299: 2, 300: 2, 301: 5, 306: 8",
            "0: 2, 1: 4, 84: 1, 87: 2, 142: 1, 210: 1, 213: 1, 273: 6, 274: 1, 286: 5, 294: 12, 299: 5, 300: 2, 301: 9, 306: 5",
            "0: 4, 1: 1, 3: 1, 84: 1, 87: 2, 142: 1, 148: 1, 210: 1, 213: 1, 273: 6, 274: 2, 286: 4, 294: 12, 299: 7, 301: 10, 304: 1, 306: 6",
            "0: 2, 1: 4, 84: 1, 87: 2, 142: 1, 210: 1, 213: 1, 273: 6, 274: 1, 286: 5, 294: 12, 299: 5, 300: 2, 301: 9, 306: 5",
            "0: 1, 4: 2, 17: 2, 48: 1, 136: 3, 180: 1, 181: 1, 183: 2, 273: 21, 274: 2, 278: 1, 286: 7, 294: 9, 301: 9, 302: 1, 306: 14",
            "0: 2, 1: 4, 84: 1, 87: 2, 142: 1, 210: 1, 213: 1, 273: 6, 274: 1, 286: 5, 294: 12, 299: 5, 300: 2, 301: 9, 306: 5",
            "0: 4, 1: 1, 3: 1, 84: 1, 87: 2, 142: 1, 148: 1, 210: 1, 213: 1, 273: 6, 274: 2, 286: 4, 294: 12, 299: 7, 301: 10, 304: 1, 306: 6",
            "0: 1, 4: 2, 17: 2, 48: 1, 136: 3, 180: 1, 181: 1, 183: 2, 273: 21, 274: 2, 278: 1, 286: 7, 294: 9, 301: 9, 302: 1, 306: 14",
            "1: 3, 2: 1, 17: 2, 87: 2, 97: 2, 99: 2, 126: 2, 210: 1, 213: 1, 273: 6, 274: 1, 286: 1, 294: 13, 299: 6, 301: 10, 304: 1, 306: 7"
        ]
        fps_ref = [fp.split(",") for fp in fps_ref]
        fps_mat_ref = np.zeros((len(fps_ref), 307))
        for i in range(len(fps_ref)):
            for fp in fps_ref[i]:
                idx, cnt = fp.split(":")
                fps_mat_ref[i, int(idx)] = int(cnt)

        df_features = candidates.get_molecule_features_by_molecule_id(molecule_ids, "substructure_count", True)
        feature_matrix = candidates.get_molecule_features_by_molecule_id(molecule_ids, "substructure_count", False)

        np.testing.assert_equal(fps_mat_ref, feature_matrix)
        np.testing.assert_equal(feature_matrix, df_features.iloc[:, 1:].values)
        self.assertListEqual(list(molecule_ids), df_features["identifier"].to_list())

        for rep in range(100):
            rnd_idc = np.random.RandomState(rep).permutation(np.arange(len(molecule_ids)))

            # Feature matrix
            feature_matrix_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "substructure_count")
            np.testing.assert_array_equal(feature_matrix[rnd_idc], feature_matrix_shf)

            # Dataframe
            df_features_shf = candidates.get_molecule_features_by_molecule_id(
                tuple(molecule_ids[i] for i in rnd_idc), "substructure_count", return_dataframe=True)
            np.testing.assert_equal(feature_matrix[rnd_idc], df_features_shf.iloc[:, 1:].values)
            self.assertListEqual([molecule_ids[i] for i in rnd_idc], df_features_shf["identifier"].to_list())

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = CandSQLiteDB_Bach2020(db_fn=BACH2020_DB_FN, molecule_identifier="inchikey")

        scores = candidates.get_ms_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_ensure_feature_is_available(self):
        candidates = CandSQLiteDB_Bach2020(db_fn=BACH2020_DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_feature_is_available("bla")
            candidates._ensure_feature_is_available(None)
            candidates._ensure_feature_is_available("")

        self.assertEqual("fingerprints_meta", candidates._ensure_feature_is_available("substructure_count"))
        self.assertEqual("fingerprints_meta", candidates._ensure_feature_is_available("iokr_fps__positive"))

    def test_ensure_molecule_identifier_is_available(self):
        candidates = CandSQLiteDB_Bach2020(db_fn=BACH2020_DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_molecule_identifier_is_available("inchistr")
            candidates._ensure_molecule_identifier_is_available(None)
            candidates._ensure_molecule_identifier_is_available("")

        candidates._ensure_molecule_identifier_is_available("inchi")
        candidates._ensure_molecule_identifier_is_available("inchikey")

    def test_parallel_access(self):
        self.skipTest("Too much memory for gh-actions.")

        candidates = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=False)
        molecule_ids = tuple([
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "BFFTVFNQHIJUQT-MRXNPFEDSA-N",
            "ATFVEXNQLNTOHV-CYBMUJFWSA-N",
            "BFFTVFNQHIJUQT-INIZCTEOSA-N",
            "CEJQCDWBMMEZFJ-NKFKGCMQSA-N",
            "BFFTVFNQHIJUQT-UHFFFAOYSA-N",
            "ATFVEXNQLNTOHV-ZDUSSCGKSA-N",
            "CEJQCDWBMMEZFJ-UHFFFAOYSA-N",
            "AZVUYIUQQIFCQK-UHFFFAOYSA-N"
        ])

        res = Parallel(n_jobs=4)(
            self._get_molecule_features_by_molecule_id(candidates, molecule_ids, "iokr_fps__positive")
            for _ in range(10000))

        for i in range(len(res)):
            np.testing.assert_array_equal(res[0], res[i])


class TestRandomSubsetCandidateSQLiteDB_Massbank(unittest.TestCase):
    def test_get_ms_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952",
                                                         "molecule_id": "CKEXCBVNKRHAMX"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False)

        # MetFrag Scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(102, len(df_scores))

        # Constant candidate scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="CONST_MS_SCORE")
        self.assertEqual(102, len(df_scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "KW13980462",
                                                         "molecule_id": "YPLYFEUBZLLLIY-UHFFFAOYSA-N"})

        # Enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=12, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)

        # MetFrag Scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(12, len(df_scores))

        # Constant candidate scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="CONST_MS_SCORE")
        self.assertEqual(12, len(df_scores))

        # --------------------

        # Enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=100000, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey",
            include_correct_candidate=True
        )

        # MetFrag Scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="metfrag__norm_after_merge")
        self.assertEqual(3834, len(df_scores))

        # Constant candidate scores
        df_scores = candidates.get_ms_scores(spectrum, ms_scorer="CONST_MS_SCORE")
        self.assertEqual(3834, len(df_scores))

    def test_get_labelspace(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "KW13980462",
                                                         "molecule_id": "YPLYFEUBZLLLIY-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=4000, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey", include_correct_candidate=False)
        self.assertEqual(3834, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "BS40569952",
                                                         "molecule_id": "CKEXCBVNKRHAMX"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "AU89649268",
                                                         "molecule_id": "SKHXRNHSZTXSLP-UKTHLTGXSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)

        scores = candidates.get_ms_scores(spectrum, "sirius__sd__correct_mf", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "sirius_fps", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "LU53695978"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "SM51162731"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=250, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(176, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=250, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(165, candidates.get_n_cand(spectrum))

    def test_get_total_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "SM39129337"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=1032, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(971, candidates.get_n_total_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(636, candidates.get_n_total_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "AU88550178"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=5, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(3934, candidates.get_n_total_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Massbank(
            number_of_candidates=102, db_fn=MASSBANK_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1999, candidates.get_n_total_cand(spectrum))


class TestRandomSubsetCandidateSQLiteDB(unittest.TestCase):
    def test_get_labelspace(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)

        scores = candidates.get_ms_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_get_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

    def test_get_total_number_of_candidates(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=1032, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(2233, candidates.get_n_total_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1918, candidates.get_n_total_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=5, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_total_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandSQLiteDB_Bach2020(
            number_of_candidates=102, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_total_cand(spectrum))


class TestABCCandSQLiteDB(unittest.TestCase):
    def test_normalize_scores(self):
        # All scores are negative
        for rep in range(30):
            _rs = np.random.RandomState(rep)
            scores = - _rs.random(_rs.randint(1, 50))
            c1, c2 = ABCCandSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            scores_norm = ABCCandSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(
                ((np.sort(scores)[1] + np.abs(np.min(scores))) / 1000) / (np.max(scores) + np.abs(np.min(scores))),
                np.min(scores_norm))

        # All scores are positive
        for rep in range(20):
            _rs = np.random.RandomState(rep)
            scores = _rs.random(_rs.randint(1, 50))
            c1, c2 = ABCCandSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            scores_norm = ABCCandSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(np.min(scores) / np.max(scores), np.min(scores_norm))

        # All scores are negative and positive
        for rep in range(20):
            _rs = np.random.RandomState(rep)
            scores = _rs.random(_rs.randint(1, 50)) - 0.5
            c1, c2 = ABCCandSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            self.assertEqual(c1, np.abs(np.min(scores)))
            self.assertEqual(c2, np.sort(scores + c1)[1] / 1000)
            scores_norm = ABCCandSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(c2 / np.max(scores + c1), np.min(scores_norm))

    def test_normalize_scores_border_cases(self):
        # All scores are equal
        for n_cand in [1, 30]:
            for val in [-1.1, -0.9, 0, 0.1, 2]:
                scores = np.full(n_cand, fill_value=val)
                c1, c2 = ABCCandSQLiteDB.get_normalization_parameters_c1_and_c2(scores)

                if val >= 0:
                    self.assertEqual(0.0, c1)
                else:
                    self.assertEqual(np.abs(val), c1)
                self.assertEqual(c2, 1e-6)
                np.testing.assert_array_equal(np.ones_like(scores), ABCCandSQLiteDB.normalize_scores(scores, c1, c2))


class TestSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.spectra_ids = ["Challenge-016", "Challenge-017", "Challenge-018", "Challenge-019", "Challenge-001"]
        self.gt_labels = [
            "FGXWKSZFVQUSTL-UHFFFAOYSA-N",
            "ZZORFUFYDOWNEF-UHFFFAOYSA-N",
            "FUYLLJCBCKRIAL-UHFFFAOYSA-N",
            "SUBDBMMJDZJVOS-UHFFFAOYSA-N",
            "UWPJYQYRSWYIGZ-UHFFFAOYSA-N"
        ]
        self.rts = np.random.RandomState(len(self.spectra_ids)).randint(low=1, high=21, size=len(self.spectra_ids))
        self.spectra = [
            Spectrum(
                np.array([]), np.array([]),
                {"spectrum_id": spectrum_id, "retention_time": rt, "molecule_identifier": mol_id}
            )
            for spectrum_id, rt, mol_id in zip(self.spectra_ids, self.rts, self.gt_labels)
        ]
        self.sequence = Sequence(spectra=self.spectra, candidates=CandSQLiteDB_Bach2020(BACH2020_DB_FN))
        self.lsequence = LabeledSequence(
            spectra=self.spectra, candidates=CandSQLiteDB_Bach2020(BACH2020_DB_FN), labels=self.gt_labels
        )

        self.sorted_sequence = Sequence(
            spectra=self.spectra, candidates=CandSQLiteDB_Bach2020(BACH2020_DB_FN), sort_sequence_by_rt=True
        )

        self.sorted_lsequence = LabeledSequence(
            spectra=self.spectra, candidates=CandSQLiteDB_Bach2020(BACH2020_DB_FN), sort_sequence_by_rt=True,
            label_key="molecule_identifier"
        )

    def test_iterator(self):
        # normal sequence
        c = 0
        for spec in self.sequence:
            self.assertIsInstance(spec, Spectrum)
            c += 1
        self.assertEqual(c, len(self.sequence))

        # labeled sequence
        c = 0
        for spec, label in self.lsequence:
            self.assertIsInstance(spec, Spectrum)
            self.assertIsInstance(label, (int, str))
            c += 1
        self.assertEqual(c, len(self.lsequence))

    def test_inchikeys_added_to_label_space(self):
        self.assertIsInstance(self.sequence.get_labelspace(0, return_inchikeys=True), dict)
        self.assertIn("inchikey", self.sequence.get_labelspace(0, return_inchikeys=True))
        self.assertIn("inchikey1", self.sequence.get_labelspace(0, return_inchikeys=True))
        self.assertIn("molecule_identifier", self.sequence.get_labelspace(0, return_inchikeys=True))
        self.assertEqual(2233, len(self.sequence.get_labelspace(0, return_inchikeys=True)["inchikey"]))

    def test_sorted_sequences(self):
        rt_before = - np.inf
        for s in range(len(self.sorted_sequence)):
            self.assertTrue(rt_before < self.sorted_sequence.get_retention_time(s))
            rt_before = self.sorted_sequence.get_retention_time(s)

    def test_sorted_labeled_sequences(self):
        # check RTs
        rt_before = - np.inf
        for s in range(len(self.sorted_lsequence)):
            self.assertTrue(rt_before < self.sorted_lsequence.get_retention_time(s))
            rt_before = self.sorted_lsequence.get_retention_time(s)

        # Check that the labels are correctly ordered
        for s in range(len(self.sorted_lsequence)):
            self.assertEqual(
                self.gt_labels[self.spectra_ids.index(self.sorted_lsequence.spectra[s].get("spectrum_id"))],
                self.sorted_lsequence.labels[s]
            )

    def test_get_label_loss(self):
        # Zero-One loss based on the molecule labels (e.g. InChIKey)
        for s in range(5):
            ll = self.lsequence.get_label_loss(zeroone_loss, "MOL_ID", s)
            self.assertEqual(1, np.sum(ll == 0))
            self.assertEqual(0, ll[self.lsequence.get_index_of_correct_structure(s)])

    def test_get_number_of_candidates(self):
        n_cand = self.sequence.get_n_cand()
        self.assertEqual(len(self.spectra_ids), len(n_cand))
        self.assertEqual(2233, n_cand[0])
        self.assertEqual(1130, n_cand[1])
        self.assertEqual(62,   n_cand[2])
        self.assertEqual(5784, n_cand[3])
        self.assertEqual(459,  n_cand[4])

    def test_get_label_space(self):
        # Get the label space
        labelspace = self.sequence.get_labelspace()
        self.assertEqual(len(self.spectra_ids), len(labelspace))

        # Compare number of candidates
        for ls, n in zip(labelspace, self.sequence.get_n_cand()):
            self.assertEqual(n, len(ls))

    def test_get_sign_delta_t(self):
        _rt_diff_signs = np.empty((len(self.spectra_ids), len(self.spectra_ids)))
        for r in range(_rt_diff_signs.shape[0]):
            for c in range(_rt_diff_signs.shape[1]):
                _rt_diff_signs[r, c] = np.sign(self.rts[r] - self.rts[c])

        np.testing.assert_equal(_rt_diff_signs, self.sequence._rt_diff_signs)

        for rep in range(10):
            G = nx.generators.trees.random_tree(len(self.spectra_ids))

            sign_delta_t_ref = np.empty(len(G.edges))
            for idx, (s, t) in enumerate(G.edges):
                sign_delta_t_ref[idx] = np.sign(self.rts[s] - self.rts[t])

            np.testing.assert_equal(sign_delta_t_ref, self.sequence.get_sign_delta_t(G))


class TestSequenceSample(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect("file:" + BACH2020_DB_FN + "?mode=ro", uri=True)

        # Read in spectra and labels
        res = pd.read_sql_query("SELECT spectrum, molecule, rt, challenge FROM challenges_spectra "
                                "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum", con=self.db)
        self.spectra = [Spectrum(np.array([]), np.array([]),
                                 {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg})
                        for (spec_id, rt, chlg) in zip(res["spectrum"], res["rt"], res["challenge"])]
        self.labels = res["molecule"].to_list()

        self.db.close()

    def test_sequence_generation(self):
        # Generate sequence sample
        N = 100
        L_min = 10
        seq_sample = SequenceSample(self.spectra, self.labels, None, N=N, L_min=L_min, random_state=201)
        self.assertEqual(N, len(seq_sample))
        self.assertTrue(all([len(seq) == L_min for seq in seq_sample]))

        # All spectra of one sequence are belonging to the same dataset
        for seq in seq_sample:
            self.assertTrue(all([seq.spectra[0].get("dataset") == seq.spectra[s].get("dataset")
                                 for s in range(len(seq))]))

    def test_train_test_splitting(self):
        # Generate sequence sample
        seq_sample = SequenceSample(self.spectra, self.labels, None, N=31, L_min=20, random_state=789)

        # Get a train test split
        train_seq, test_seq = seq_sample.get_train_test_split()

        # Check length
        self.assertEqual(6, len(test_seq))
        self.assertEqual(25, len(train_seq))

        # No intersection of molecules between training and test sequences
        for i, j in it.product(test_seq, train_seq):
            self.assertTrue(len(set(i.labels) & set(j.labels)) == 0)

    def test_sequence_specific_candidate_set(self):
        # self.skipTest("Needs to be implemented.")
        # Generate sequence sample
        number_of_candidates = 5
        seq_sample = SequenceSample(
            self.spectra, self.labels,
            RandomSubsetCandSQLiteDB_Bach2020(
                number_of_candidates=number_of_candidates, db_fn=BACH2020_DB_FN, molecule_identifier="inchikey",
                include_correct_candidate=False, random_state=1020, init_with_open_db_conn=False),
            N=31, L_min=20,
            random_state=484
        )

        # Get a train test split
        train_seq, test_seq = seq_sample.get_train_test_split(use_sequence_specific_candidates_for_training=True)

        # Ensure that for the training set each sequence is associated with a different candidate set DB.
        for i in range(len(train_seq)):
            for j in range(len(train_seq)):
                if i == j:
                    continue

                # Candidate sets must point to different objects
                self.assertIsNot(train_seq[i].candidates, train_seq[j].candidates)

                # Get spectrum ids
                li = [spec.get("spectrum_id") for spec in train_seq[i].spectra]
                lj = [spec.get("spectrum_id") for spec in train_seq[j].spectra]

                # Go over the intersection of the spectrum ids (if any)
                for spec_id in (set(li) & set(lj)):
                    idx_i = li.index(spec_id)
                    idx_j = lj.index(spec_id)

                    with train_seq[i].candidates, train_seq[j].candidates:
                        # Size of the candidate (sub-) set must be equal
                        self.assertEqual(train_seq[i].get_n_total_cand(idx_i), train_seq[j].get_n_total_cand(idx_j))
                        self.assertEqual(train_seq[i].get_n_cand(idx_i), train_seq[j].get_n_cand(idx_j))

                        if train_seq[i].get_n_total_cand(idx_i) <= number_of_candidates:
                            # All candidates must be selected so the subsets must be equal
                            self.assertEqual(train_seq[i].get_labelspace(idx_i), train_seq[j].get_labelspace(idx_j))
                        elif number_of_candidates < train_seq[i].get_n_total_cand(idx_i) < (number_of_candidates + 5):
                            # The total number of candidates is very small, it is likely that the same candidate sets
                            # are sampled even randomly.
                            pass
                        else:
                            self.assertNotEqual(train_seq[i].get_labelspace(idx_i), train_seq[j].get_labelspace(idx_j))

        for i in range(len(test_seq)):
            for j in range(len(test_seq)):
                if i == j:
                    continue

                # Candidate sets must point to different objects
                self.assertIs(test_seq[i].candidates, test_seq[j].candidates)


class TestPerformanceDifferenceBetweenBach2020AndMassBank(unittest.TestCase):
    def test_loading_molecule_features_by_id(self):
        n_rep = 25

        # --------
        # MassBank
        # --------

        candidates_mb = CandSQLiteDB_Massbank(MASSBANK_DB_FN, molecule_identifier="inchikey")

        # BINARY

        t_mb_binary = 0.0
        for rep in range(n_rep):
            res = candidates_mb.db.execute(
                "SELECT inchikey FROM fingerprints_data__sirius_fps fd \
                 INNER JOIN molecules m ON fd.molecule = m.cid \
                 ORDER BY random() \
                 LIMIT 100"
            )

            start = time.time()
            fps = candidates_mb.get_molecule_features_by_molecule_id([row[0] for row in res], "sirius_fps")
            t_mb_binary += (time.time() - start)

        print("MassBank (binary): %.3fs" % (t_mb_binary / n_rep))

        # COUNT

        t_mb_count = 0.0
        for rep in range(n_rep):
            res = candidates_mb.db.execute(
                "SELECT inchikey FROM fingerprints_data__FCFP__count__all fd \
                 INNER JOIN molecules m ON fd.molecule = m.cid \
                 ORDER BY random() \
                 LIMIT 200"
            )

            start = time.time()
            candidates_mb.get_molecule_features_by_molecule_id([row[0] for row in res], "FCFP__count__all")
            t_mb_count += (time.time() - start)

        print("MassBank (count): %.3fs" % (t_mb_count / n_rep))

        # --------
        # Bach2020
        # --------
        candidates_bach = CandSQLiteDB_Bach2020(BACH2020_DB_FN, molecule_identifier="inchikey")

        t_bach = 0.0
        for rep in range(n_rep):
            res = candidates_bach.db.execute(
                "SELECT inchikey FROM molecules \
                 ORDER BY random() \
                 LIMIT 200"
            )

            start = time.time()
            candidates_bach.get_molecule_features_by_molecule_id([row[0] for row in res], "substructure_count")
            t_bach += (time.time() - start)

        print("Bach2020: %.3fs" % (t_bach / n_rep))


class TestBugsAndWiredStuff(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect("file:" + BACH2020_DB_FN + "?mode=ro", uri=True)

        # Read in spectra and labels
        res = pd.read_sql_query("SELECT spectrum, molecule, rt, challenge FROM challenges_spectra "
                                "   INNER JOIN spectra s ON s.name = challenges_spectra.spectrum", con=self.db)
        self.spectra = [Spectrum(np.array([]), np.array([]),
                                 {"spectrum_id": spec_id, "retention_time": rt, "dataset": chlg})
                        for (spec_id, rt, chlg) in zip(res["spectrum"], res["rt"], res["challenge"])]
        self.labels = res["molecule"].to_list()

        self.db.close()

    def test_different_candidates_sets_between_ms2scorer(self):
        from sklearn.model_selection import GroupShuffleSplit
        _, test = next(
            GroupShuffleSplit(test_size=0.2, random_state=10).split(np.arange(len(self.spectra)), groups=self.labels))

        test_sequences_metfrag = SequenceSample(
            [self.spectra[idx] for idx in test], [self.labels[idx] for idx in test],
            CandSQLiteDB_Bach2020(db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1", init_with_open_db_conn=True),
            N=50, L_min=30, L_max=50, random_state=19, ms_scorer="MetFrag_2.4.5__8afe4a14")

        test_sequences_iokr = SequenceSample(
            [self.spectra[idx] for idx in test], [self.labels[idx] for idx in test],
            CandSQLiteDB_Bach2020(db_fn=BACH2020_DB_FN, molecule_identifier="inchikey1", init_with_open_db_conn=True),
            N=50, L_min=30, L_max=50, random_state=19, ms_scorer="IOKR__696a17f3")

        for idx in range(len(test_sequences_metfrag)):
            self.assertListEqual(test_sequences_metfrag[idx].get_labelspace(),
                                 test_sequences_iokr[idx].get_labelspace())


class TestRandomSpanningTrees(unittest.TestCase):
    class DummySequence(object):
        def __init__(self):
            self.elements = ["F", "A", "B", "C", "D", "E"]
            self.rts = np.random.RandomState(len(self.elements)).rand(len(self.elements))

        def get_retention_time(self, s: int):
            return self.rts[s]

        def __iter__(self):
            return self.elements.__iter__()

        def __len__(self):
            return self.elements.__len__()

    def test_random_seed_leads_to_different_trees(self):
        RST = SpanningTrees(self.DummySequence(), n_trees=4, random_state=10)

        for s in range(len(RST)):
            for t in range(len(RST)):
                if s == t:
                    self.assertEqual(RST[s].edges, RST[t].edges)
                else:
                    self.assertNotEqual(RST[s].edges, RST[t].edges)


if __name__ == '__main__':
    unittest.main()
