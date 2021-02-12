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

from matchms.Spectrum import Spectrum
from joblib import Parallel, delayed
from scipy.stats import rankdata

from ssvm.data_structures import SequenceSample, CandidateSQLiteDB, RandomSubsetCandidateSQLiteDB
from ssvm.data_structures import Sequence, HigherRankedCandidatesSQLiteDB

DB_FN = "/home/bach/Documents/doctoral/projects/local_casmi_db/db/use_inchis/DB_LATEST.db"


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
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
        self.assertEqual(2233, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(1918, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

    def test_get_ms2_scores(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is IOKR
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="IOKR__696a17f3")
        self.assertEqual(2233, len(scores))
        self.assertEqual(0.006426196626211542, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(2233, len(scores))
        self.assertEqual(0.0028105936908001407, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")

        # MS2 Scorer is IOKR
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="IOKR__696a17f3")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.7441697188705315, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

        # MS2 Scorer is MetFrag
        scores = candidates.get_ms2_scores(spectrum, ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(16, len(scores))
        self.assertEqual(0.18165470462229025, np.min(scores))
        self.assertEqual(1.0, np.max(scores))

    def test_get_molecule_features(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-019"})
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey1")

        # IOKR features
        fps = candidates.get_molecule_features(spectrum, features="iokr_fps__positive")
        self.assertEqual((5103, 7936), fps.shape)
        self.assertTrue(np.all(np.isin(fps, [0, 1])))

        # Substructure features
        fps = candidates.get_molecule_features(spectrum, features="substructure_count")
        self.assertEqual((5103, 307), fps.shape)
        self.assertTrue(np.all(fps >= 0))

    def test_get_molecule_feature_by_molecule_id__repeated_ids(self):
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
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

    def test_get_molecule_feature_by_molecule_id__ids_as_tuple(self):
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")

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
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey")
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
        candidates = CandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey")

        scores = candidates.get_ms2_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_ensure_feature_is_available(self):
        candidates = CandidateSQLiteDB(db_fn=DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_feature_is_available("bla")
            candidates._ensure_feature_is_available(None)
            candidates._ensure_feature_is_available("")

        candidates._ensure_feature_is_available("substructure_count")
        candidates._ensure_feature_is_available("iokr_fps__positive")

    def test_ensure_molecule_identifier_is_available(self):
        candidates = CandidateSQLiteDB(db_fn=DB_FN)

        with self.assertRaises(ValueError):
            candidates._ensure_molecule_identifier_is_available("inchistr")
            candidates._ensure_molecule_identifier_is_available(None)
            candidates._ensure_molecule_identifier_is_available("")

        candidates._ensure_molecule_identifier_is_available("inchi")
        candidates._ensure_molecule_identifier_is_available("inchikey")

    def test_parallel_access(self):
        candidates = CandidateSQLiteDB(DB_FN, molecule_identifier="inchikey", init_with_open_db_conn=False)
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

        print("LOAD")
        res = Parallel(n_jobs=4)(
            self._get_molecule_features_by_molecule_id(candidates, molecule_ids, "iokr_fps__positive")
            for _ in range(10000))

        print("TEST")
        for i in range(len(res)):
            np.testing.assert_array_equal(res[0], res[i])


class TestRandomSubsetCandidateSQLiteDB(unittest.TestCase):
    def test_get_labelspace(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False)
        self.assertEqual(102, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(np.unique(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=True)
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
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True)

        scores = candidates.get_ms2_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
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
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(102, candidates.get_n_cand(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX030601"})

        # Molecule identifier: inchikey
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

        # Molecule identifier: inchikey1
        candidates = RandomSubsetCandidateSQLiteDB(
            number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey1")
        self.assertEqual(16, candidates.get_n_cand(spectrum))

    def test_normalize_scores(self):
        # All scores are negative
        for rep in range(30):
            _rs = np.random.RandomState(rep)
            scores = - _rs.random(_rs.randint(1, 50))
            c1, c2 = CandidateSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            scores_norm = CandidateSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(
                ((np.sort(scores)[1] + np.abs(np.min(scores))) / 10) / (np.max(scores) + np.abs(np.min(scores))),
                np.min(scores_norm))

        # All scores are positive
        for rep in range(20):
            _rs = np.random.RandomState(rep)
            scores = _rs.random(_rs.randint(1, 50))
            c1, c2 = CandidateSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            scores_norm = CandidateSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(np.min(scores) / np.max(scores), np.min(scores_norm))

        # All scores are negative and positive
        for rep in range(20):
            _rs = np.random.RandomState(rep)
            scores = _rs.random(_rs.randint(1, 50)) - 0.5
            c1, c2 = CandidateSQLiteDB.get_normalization_parameters_c1_and_c2(scores)
            self.assertEqual(c1, np.abs(np.min(scores)))
            self.assertEqual(c2, np.sort(scores + c1)[1] / 10)
            scores_norm = CandidateSQLiteDB.normalize_scores(scores, c1, c2)
            np.testing.assert_array_equal(rankdata(scores, method="ordinal"), rankdata(scores_norm, method="ordinal"))
            self.assertEqual(1.0, np.max(scores_norm))
            self.assertAlmostEqual(c2 / np.max(scores + c1), np.min(scores_norm))

    def test_normalize_scores_border_cases(self):
        # All scores are equal
        for n_cand in [1, 30]:
            for val in [-1.1, -0.9, 0, 0.1, 2]:
                scores = np.full(n_cand, fill_value=val)
                c1, c2 = CandidateSQLiteDB.get_normalization_parameters_c1_and_c2(scores)

                if val >= 0:
                    self.assertEqual(0.0, c1)
                else:
                    self.assertEqual(np.abs(val), c1)
                self.assertEqual(c2, 1e-6)
                np.testing.assert_array_equal(np.ones_like(scores), CandidateSQLiteDB.normalize_scores(scores, c1, c2))


class TestHigherRankedCandidatesSQLiteDB(unittest.TestCase):
    def test_get_labelspace(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=25, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=False,
            ms2scorer="IOKR__696a17f3")
        self.assertEqual(25, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=25, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True,
            ms2scorer="IOKR__696a17f3")
        self.assertEqual(25, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=25, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=False,
            ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(2, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=25, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True,
            ms2scorer="MetFrag_2.4.5__8afe4a14")
        self.assertEqual(2, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=200, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=False,
            ms2scorer="IOKR__696a17f3")
        self.assertEqual(184, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))

        # Enforce correct structure to be present
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=200, db_fn=DB_FN, molecule_identifier="inchikey1", include_correct_candidate=True,
            ms2scorer="IOKR__696a17f3")
        self.assertEqual(184, len(candidates.get_labelspace(spectrum)))
        self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
        self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

    def test_all_outputs_are_sorted_equally(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL-UHFFFAOYSA-N"})

        # Do not enforce the ground truth structure to be in the candidate set
        candidates = HigherRankedCandidatesSQLiteDB(
            max_number_of_candidates=102, db_fn=DB_FN, molecule_identifier="inchikey", include_correct_candidate=True,
            ms2scorer="MetFrag_2.4.5__8afe4a14", score_correction_factor=0.95)

        scores = candidates.get_ms2_scores(spectrum, "MetFrag_2.4.5__8afe4a14", return_dataframe=True)
        fps = candidates.get_molecule_features(spectrum, "iokr_fps__positive", return_dataframe=True)
        labspace = candidates.get_labelspace(spectrum)

        self.assertEqual(sorted(labspace), labspace)
        self.assertEqual(labspace, scores["identifier"].to_list())
        self.assertEqual(labspace, fps["identifier"].to_list())

    def test_different_score_factors__IOKR(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        for max_n_cand in [10, 20, 100, np.inf]:
            for fac, n_cand in zip([0, 0.95, 1.0], [1918, 184, 53]):
                # Enforce correct structure to be present
                candidates = HigherRankedCandidatesSQLiteDB(
                    max_number_of_candidates=max_n_cand, score_correction_factor=fac, ms2scorer="IOKR__696a17f3",
                    include_correct_candidate=True, db_fn=DB_FN, molecule_identifier="inchikey1")

                self.assertEqual(np.minimum(max_n_cand, n_cand), len(candidates.get_labelspace(spectrum)))
                self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
                self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX034002",
                                                         "molecule_id": "InChI=1S/C9H14N2O2S/c1-8-4-6-9(7-5-8)10-14(12,13)11(2)3/h4-7,10H,1-3H3"})

        for max_n_cand in [10, 20, 100, np.inf]:
            for fac, n_cand in zip([0, 0.95, 1.0], [525, 396, 316]):
                # Enforce correct structure to be present
                candidates = HigherRankedCandidatesSQLiteDB(
                    max_number_of_candidates=max_n_cand, score_correction_factor=fac, ms2scorer="IOKR__696a17f3",
                    include_correct_candidate=True, db_fn=DB_FN, molecule_identifier="inchi")

                self.assertEqual(np.minimum(max_n_cand, n_cand), len(candidates.get_labelspace(spectrum)))
                self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
                self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

    def test_different_score_factors__MetFrag(self):
        # ----------
        # SPECTRUM 1
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "Challenge-016",
                                                         "molecule_id": "FGXWKSZFVQUSTL"})

        for max_n_cand in [1, 10, 20, 100, np.inf]:
            for fac, n_cand in zip([0, 0.95, 1.0], [1918, 2, 2]):
                # Enforce correct structure to be present
                candidates = HigherRankedCandidatesSQLiteDB(
                    max_number_of_candidates=max_n_cand, score_correction_factor=fac,
                    ms2scorer="MetFrag_2.4.5__8afe4a14", include_correct_candidate=True, db_fn=DB_FN,
                    molecule_identifier="inchikey1")

                self.assertEqual(np.minimum(max_n_cand, n_cand), len(candidates.get_labelspace(spectrum)))
                self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
                self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))

        # ----------
        # SPECTRUM 2
        # ----------
        spectrum = Spectrum(np.array([]), np.array([]), {"spectrum_id": "EAX034002",
                                                         "molecule_id": "InChI=1S/C9H14N2O2S/c1-8-4-6-9(7-5-8)10-14(12,13)11(2)3/h4-7,10H,1-3H3"})

        for max_n_cand in [10, 20, 100, np.inf]:
            for fac, n_cand in zip([0, 0.75, 0.95, 1.0], [525, 99, 76, 76]):
                # Enforce correct structure to be present
                candidates = HigherRankedCandidatesSQLiteDB(
                    max_number_of_candidates=max_n_cand, score_correction_factor=fac,
                    ms2scorer="MetFrag_2.4.5__8afe4a14", include_correct_candidate=True, db_fn=DB_FN,
                    molecule_identifier="inchi")

                self.assertEqual(np.minimum(max_n_cand, n_cand), len(candidates.get_labelspace(spectrum)))
                self.assertEqual(candidates.get_n_cand(spectrum), len(set(candidates.get_labelspace(spectrum))))
                self.assertIn(spectrum.metadata["molecule_id"], candidates.get_labelspace(spectrum))


class TestSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.spectra_ids = ["Challenge-016", "Challenge-017", "Challenge-018", "Challenge-019", "Challenge-001"]
        self.rts = np.random.RandomState(len(self.spectra_ids)).randint(low=1, high=21, size=len(self.spectra_ids))
        self.spectra = [Spectrum(np.array([]), np.array([]), {"spectrum_id": spectrum_id, "retention_time": rt})
                        for spectrum_id, rt in zip(self.spectra_ids, self.rts)]
        self.sequence = Sequence(spectra=self.spectra, candidates=CandidateSQLiteDB(DB_FN))

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
        self.db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

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
        self.assertTrue(all([len(ss) == L_min for ss in seq_sample]))

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


class TestBugsAndWiredStuff(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect("file:" + DB_FN + "?mode=ro", uri=True)

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
            CandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey1", init_with_open_db_conn=True),
            N=50, L_min=30, L_max=50, random_state=19, ms2scorer="MetFrag_2.4.5__8afe4a14")

        test_sequences_iokr = SequenceSample(
            [self.spectra[idx] for idx in test], [self.labels[idx] for idx in test],
            CandidateSQLiteDB(db_fn=DB_FN, molecule_identifier="inchikey1", init_with_open_db_conn=True),
            N=50, L_min=30, L_max=50, random_state=19, ms2scorer="IOKR__696a17f3")

        for idx in range(len(test_sequences_metfrag)):
            self.assertListEqual(test_sequences_metfrag[idx].get_labelspace(),
                                 test_sequences_iokr[idx].get_labelspace())


if __name__ == '__main__':
    unittest.main()
