####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
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
import unittest
import numpy as np
import time
import opt_einsum as oe

from joblib import Parallel, delayed


class TestVectorization(unittest.TestCase):
    """
    Test descriptions
    -----------------

    test_1: We empirically validate the different vectorizations of expression I (see Eqs. (37 - 40))
    test_2:
    """
    def test_1(self):
        C = 2
        N = 1000
        nE = 25
        L_jbss = []
        L_jbtt = []
        L_jbst = []
        L_jbts = []
        sign_Delta_tj = []
        for j in range(N):
            L_jbss.append(np.random.RandomState(j * 5 + 1).rand(nE))
            L_jbtt.append(np.random.RandomState(j * 5 + 2).rand(nE))
            L_jbst.append(np.random.RandomState(j * 5 + 3).rand(nE))
            L_jbts.append(np.random.RandomState(j * 5 + 4).rand(nE))
            sign_Delta_tj.append(np.sign(np.random.RandomState(j * 5 + 4).randn(nE)))

        # Eq. (37)
        start = time.time()
        res_37 = 0.0
        for j in range(N):
            res_37 += np.sum(sign_Delta_tj[j] * (L_jbss[j] * L_jbtt[j] - L_jbst[j] * L_jbts[j]))
        res_37 *= (C / N)
        print("Eq. (37): %.5fs" % (time.time() - start))

        # Eq. (38)
        print(np.column_stack(sign_Delta_tj).shape)
        start = time.time()
        res_38 = C / N * np.sum(np.column_stack(sign_Delta_tj) * (np.column_stack(L_jbss) * np.column_stack(L_jbtt) -
                                                                  np.column_stack(L_jbst) * np.column_stack(L_jbts)))
        print("Eq. (38): %.5fs" % (time.time() - start))
        np.testing.assert_almost_equal(res_37, res_38)

        # Eq. (39)
        print(np.concatenate(sign_Delta_tj).shape)
        start = time.time()
        res_39 = C / N * np.sum(np.concatenate(sign_Delta_tj) * (
                np.concatenate(L_jbss) * np.concatenate(L_jbtt) - np.concatenate(L_jbst) * np.concatenate(L_jbts)))
        print("Eq. (39): %.5fs" % (time.time() - start))
        np.testing.assert_almost_equal(res_37, res_39)

        # Eq. (40)
        start = time.time()
        res_40 = C / N * np.concatenate(sign_Delta_tj) @ (
                np.concatenate(L_jbss) * np.concatenate(L_jbtt) - np.concatenate(L_jbst) * np.concatenate(L_jbts))
        print("Eq. (40): %.5fs" % (time.time() - start))
        np.testing.assert_almost_equal(res_37, res_40)

    def test_2(self):
        C = 2
        N = 100
        nE = 25
        nSs = 7
        nSt = 12
        L_jbss = []
        L_jbtt = []
        L_jbst = []
        L_jbts = []
        sign_Delta_tj = []
        for j in range(N):
            L_jbss.append(np.random.RandomState(j * 5 + 1).rand(nE, nSs))
            L_jbtt.append(np.random.RandomState(j * 5 + 2).rand(nE, nSt))
            L_jbst.append(np.random.RandomState(j * 5 + 3).rand(nE, nSt))
            L_jbts.append(np.random.RandomState(j * 5 + 4).rand(nE, nSs))
            sign_Delta_tj.append(np.sign(np.random.RandomState(j * 5 + 4).randn(nE)))

        # Eq. (37)
        start = time.time()
        res_37 = np.zeros((nSs, nSt))
        for s in range(nSs):
            for t in range(nSt):
                for j in range(N):
                    res_37[s, t] += np.sum(sign_Delta_tj[j] * (L_jbss[j][:, s] * L_jbtt[j][:, t] -
                                                               L_jbst[j][:, t] * L_jbts[j][:, s]))
        res_37 *= (C / N)
        print("Eq. (37):\t%.5fs" % (time.time() - start))

        # Eq. (37)
        start = time.time()
        res_37a = np.zeros((nSs, nSt))
        for j in range(N):
            res_37a += np.einsum("i...,i",
                                 L_jbss[j][:, :, np.newaxis] * L_jbtt[j][:, np.newaxis, :] -
                                 L_jbst[j][:, np.newaxis, :] * L_jbts[j][:, :, np.newaxis],
                                 sign_Delta_tj[j])
        res_37a *= (C / N)
        print("Eq. (37a):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37a.shape)
        np.testing.assert_allclose(res_37, res_37a)

    def test_3(self):
        def _fun_37c(S, L1, L2, path):
            return np.einsum("i,ij,ik", S, L1, L2, optimize=path)

        C = 2
        N = 30
        nE = 50
        nSs = 25
        nSt = 18
        L_jbss = np.random.RandomState(1).rand(N, nE, nSs)
        L_jbtt = np.random.RandomState(2).rand(N, nE, nSt)
        L_jbst = np.random.RandomState(3).rand(N, nE, nSt)
        L_jbts = np.random.RandomState(4).rand(N, nE, nSs)
        sign_Delta_tj = np.sign(np.random.RandomState(5).randn(N, nE))

        # Eq. (37)
        start = time.time()
        res_37 = np.zeros((nSs, nSt))
        for s in range(nSs):
            for t in range(nSt):
                for j in range(N):
                    res_37[s, t] += np.sum(sign_Delta_tj[j] * (L_jbss[j, :, s] * L_jbtt[j, :, t] -
                                                               L_jbst[j, :, t] * L_jbts[j, :, s]))
        res_37 *= (C / N)
        print("Eq. (37):\t%.5fs" % (time.time() - start))

        # # Eq. (37)
        start = time.time()
        res_37a = np.zeros((nSs, nSt))
        for j in range(N):
            res_37a += np.einsum("i...,i",
                                 L_jbss[j, :, :, np.newaxis] * L_jbtt[j, :, np.newaxis, :] -
                                 L_jbst[j, :, np.newaxis, :] * L_jbts[j, :, :, np.newaxis],
                                 sign_Delta_tj[j])
        res_37a *= (C / N)
        print("Eq. (37a):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37a.shape)
        np.testing.assert_allclose(res_37, res_37a)

        # Eq. (37)
        start = time.time()
        res_37b = np.einsum("ij...,ij",
                            L_jbss[:, :, :, np.newaxis] * L_jbtt[:, :, np.newaxis, :] -
                            L_jbst[:, :, np.newaxis, :] * L_jbts[:, :, :, np.newaxis],
                            sign_Delta_tj * C / N)
        print("Eq. (37b):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37b.shape)
        np.testing.assert_allclose(res_37, res_37b)

        # Eq. (37)
        start = time.time()
        res_37c = np.zeros((nSs, nSt))
        path = np.einsum_path("i,ij,ik", sign_Delta_tj[0], L_jbss[0], L_jbtt[0], optimize=True)[0]
        for j in range(N):
            A = np.einsum("i,ij,ik", sign_Delta_tj[j], L_jbss[j], L_jbtt[j], optimize=path)
            B = np.einsum("i,ij,ik", sign_Delta_tj[j], L_jbts[j], L_jbst[j], optimize=path)
            res_37c += (A - B)
        res_37c *= (C / N)
        print("Eq. (37c):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37c.shape)
        np.testing.assert_allclose(res_37, res_37c)

        # Eq. (37)
        start = time.time()
        res_37d = np.zeros((nSs, nSt))
        expr = oe.contract_expression("i,ij,ik", sign_Delta_tj[0].shape, L_jbss[0].shape, L_jbtt[0].shape)
        print(expr)
        for j in range(N):
            A = expr(sign_Delta_tj[j], L_jbss[j], L_jbtt[j])
            B = expr(sign_Delta_tj[j], L_jbts[j], L_jbst[j])
            res_37d += (A - B)
        res_37d *= (C / N)
        print("Eq. (37d):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37d.shape)
        np.testing.assert_allclose(res_37, res_37d)

        # Eq. (37)
        start = time.time()
        expr = oe.contract_expression("ij,ijk,ijl", sign_Delta_tj.shape, L_jbss.shape, L_jbtt.shape)
        print(expr)
        A = expr(sign_Delta_tj, L_jbss, L_jbtt)
        B = expr(sign_Delta_tj, L_jbts, L_jbst)
        res_37e = C / N * (A - B)
        print("Eq. (37e):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37e.shape)
        np.testing.assert_allclose(res_37, res_37e)

        # Eq. (37)
        start = time.time()
        path = np.einsum_path("i,ij,ik", sign_Delta_tj[0], L_jbss[0], L_jbtt[0], optimize=True)[0]
        res_A = Parallel(n_jobs=4)(delayed(_fun_37c)(sign_Delta_tj[j], L_jbss[j], L_jbtt[j], path) for j in range(N))
        res_B = Parallel(n_jobs=4)(delayed(_fun_37c)(sign_Delta_tj[j], L_jbts[j], L_jbst[j], path) for j in range(N))
        res_37f = C / N * (np.sum(res_A, axis=0) - np.sum(res_B, axis=0))
        print("Eq. (37f):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37f.shape)
        np.testing.assert_allclose(res_37, res_37f)

    def test_3_speed_on_large_data(self):
        def _fun_37c(S, L1, L2, path):
            return np.einsum("i,ij,ik", S, L1, L2, optimize=path)

        C = 2
        N = 5000
        nE = 30
        nSs = 200
        nSt = 200
        L_jbss = np.random.RandomState(1).rand(N, nE, nSs)
        L_jbtt = np.random.RandomState(2).rand(N, nE, nSt)
        L_jbst = np.random.RandomState(3).rand(N, nE, nSt)
        L_jbts = np.random.RandomState(4).rand(N, nE, nSs)
        sign_Delta_tj = np.sign(np.random.RandomState(5).randn(N, nE))

        # Eq. (37)
        start = time.time()
        res_37c = np.zeros((nSs, nSt))
        path = np.einsum_path("i,ij,ik", sign_Delta_tj[0], L_jbss[0], L_jbtt[0], optimize=True)[0]
        for j in range(N):
            A = np.einsum("i,ij,ik", sign_Delta_tj[j], L_jbss[j], L_jbtt[j], optimize=path)
            B = np.einsum("i,ij,ik", sign_Delta_tj[j], L_jbts[j], L_jbst[j], optimize=path)
            res_37c += (A - B)
        res_37c *= (C / N)
        print("Eq. (37c):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37c.shape)

        # Eq. (37)
        start = time.time()
        res_37d = np.zeros((nSs, nSt))
        expr = oe.contract_expression("i,ij,ik", sign_Delta_tj[0].shape, L_jbss[0].shape, L_jbtt[0].shape)
        for j in range(N):
            A = expr(sign_Delta_tj[j], L_jbss[j], L_jbtt[j])
            B = expr(sign_Delta_tj[j], L_jbts[j], L_jbst[j])
            res_37d += (A - B)
        res_37d *= (C / N)
        print("Eq. (37d):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37d.shape)

        # Eq. (37)
        start = time.time()
        expr = oe.contract_expression("ij,ijk,ijl", sign_Delta_tj, L_jbss.shape, L_jbtt.shape, constants=[0])
        A = expr(L_jbss, L_jbtt)
        B = expr(L_jbts, L_jbst)
        res_37e = C / N * (A - B)
        print("Eq. (37e):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37e.shape)
        print(expr)

        # Eq. (37)
        start = time.time()
        path = np.einsum_path("i,ij,ik", sign_Delta_tj[0], L_jbss[0], L_jbtt[0], optimize=True)[0]
        res_A = Parallel(n_jobs=4)(delayed(_fun_37c)(sign_Delta_tj[j], L_jbss[j], L_jbtt[j], path) for j in range(N))
        res_B = Parallel(n_jobs=4)(delayed(_fun_37c)(sign_Delta_tj[j], L_jbts[j], L_jbst[j], path) for j in range(N))
        res_37f = C / N * (np.sum(res_A, axis=0) - np.sum(res_B, axis=0))
        print("Eq. (37f):\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_37f.shape)

    def test_4(self):
        N = 50
        nE = 23
        nSs = 35
        nSt = 36
        nSj_min, nSj_max = 10, 52

        L_jbss = []
        L_jbtt = []
        L_jbst = []
        L_jbts = []
        alpha_j = []
        n_active = []
        for j in range(N):
            nSj = np.random.RandomState(j * 6 + 1).randint(nSj_min, nSj_max)
            L_jbss.append(np.random.RandomState(j * 6 + 2).rand(nE, nSj, nSs))
            L_jbtt.append(np.random.RandomState(j * 6 + 3).rand(nE, nSj, nSt))
            L_jbst.append(np.random.RandomState(j * 6 + 4).rand(nE, nSj, nSt))
            L_jbts.append(np.random.RandomState(j * 6 + 5).rand(nE, nSj, nSs))

            _alpha = np.random.RandomState(j * 6 + 6).randn(nSj)
            _alpha[_alpha < 0] = 0.0
            alpha_j.append(_alpha)
            n_active.append(np.sum(_alpha > 0).item())
            # print("j = %d:\t#a>0 = %d (of %d)" % (j, n_active[-1], len(_alpha)))

        sign_Delta_tj = np.sign(np.random.RandomState(10).randn(N, nE))

        start = time.time()
        res = np.zeros((nSs, nSt))
        for s in range(nSs):
            for t in range(nSt):
                for j in range(N):
                    res[s, t] += sign_Delta_tj[j] @ (L_jbss[j][:, :, s] * L_jbtt[j][:, :, t] -
                                                     L_jbst[j][:, :, t] * L_jbts[j][:, :, s]) @ alpha_j[j]
        print("Baseline:\t%.5fs" % (time.time() - start))

        # --- Using active set ---
        L_jbss_active = [L_jbss[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbtt_active = [L_jbtt[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbst_active = [L_jbst[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbts_active = [L_jbts[j][:, alpha_j[j] > 0, :] for j in range(N)]
        alpha_j_active = [alpha_j[j][alpha_j[j] > 0] for j in range(N)]

        start = time.time()
        res_a = np.zeros((nSs, nSt))
        for s in range(nSs):
            for t in range(nSt):
                for j in range(N):
                    res_a[s, t] += sign_Delta_tj[j] @ (L_jbss_active[j][:, :, s] * L_jbtt_active[j][:, :, t] -
                                                       L_jbst_active[j][:, :, t] * L_jbts_active[j][:, :, s]) \
                                   @ alpha_j_active[j]
        print("A:\t%.5fs" % (time.time() - start))
        np.testing.assert_allclose(res, res_a)

        start = time.time()
        res_b = np.zeros((nSs, nSt))
        for j in range(N):
            A = oe.contract("i,ijk,ijl,j", sign_Delta_tj[j], L_jbss_active[j], L_jbtt_active[j], alpha_j_active[j])
            B = oe.contract("i,ijk,ijl,j", sign_Delta_tj[j], L_jbts_active[j], L_jbst_active[j], alpha_j_active[j])
            res_b += (A - B)
        print("B:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_b.shape)
        np.testing.assert_allclose(res, res_b)

        S = np.repeat(np.column_stack(sign_Delta_tj), repeats=n_active, axis=1)
        M_jbss = np.concatenate(L_jbss_active, axis=1)
        M_jbtt = np.concatenate(L_jbtt_active, axis=1)
        M_jbst = np.concatenate(L_jbst_active, axis=1)
        M_jbts = np.concatenate(L_jbts_active, axis=1)
        Alphas = np.concatenate(alpha_j_active)

        start = time.time()
        expr = oe.contract_expression("ij,ijk,ijl,j", S.shape, M_jbss.shape, M_jbtt.shape, Alphas.shape)
        A = expr(S, M_jbss, M_jbtt, Alphas)
        B = expr(S, M_jbts, M_jbst, Alphas)
        res_c = A - B
        print("C:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_c.shape)
        np.testing.assert_allclose(res, res_c)

        start = time.time()
        res_d = np.zeros((nSs, nSt))
        for j in range(N):
            expr = oe.contract_expression("i,j,ijk,ijl",
                                          sign_Delta_tj[j], alpha_j_active[j],
                                          L_jbss_active[j].shape, L_jbtt_active[j].shape,
                                          constants=[0, 1])
            A = expr(L_jbss_active[j], L_jbtt_active[j])
            B = expr(L_jbts_active[j], L_jbst_active[j])
            res_d += (A - B)
        print("D:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_d.shape)
        np.testing.assert_allclose(res, res_d)
        print(expr)

        # start = time.time()
        # expr = oe.contract_expression("ij,ijk,ijl,j", S, M_jbss.shape, M_jbtt.shape, Alphas, constants=[0, 3])
        # A = expr(M_jbss, M_jbtt)
        # B = expr(M_jbts, M_jbst)
        # res_d = A - B
        # print("D:\t%.5fs" % (time.time() - start))
        # self.assertEqual((nSs, nSt), res_d.shape)
        # np.testing.assert_allclose(res, res_d)

    def test_4_speed_on_large_data(self):
        N = 3000
        nE = 30
        nSs = 200
        nSt = 200
        nSj_min, nSj_max = 5, 20

        L_jbss = []
        L_jbtt = []
        L_jbst = []
        L_jbts = []
        alpha_j = []
        n_active = []
        for j in range(N):
            nSj = np.random.RandomState(j * 6 + 1).randint(nSj_min, nSj_max)
            L_jbss.append(np.random.RandomState(j * 6 + 2).rand(nE, nSj, nSs))
            L_jbtt.append(np.random.RandomState(j * 6 + 3).rand(nE, nSj, nSt))
            L_jbst.append(np.random.RandomState(j * 6 + 4).rand(nE, nSj, nSt))
            L_jbts.append(np.random.RandomState(j * 6 + 5).rand(nE, nSj, nSs))

            _alpha = np.random.RandomState(j * 6 + 6).randn(nSj)
            _alpha[_alpha < 0] = 0.0
            alpha_j.append(_alpha)
            n_active.append(np.sum(_alpha > 0).item())
            # print("j = %d:\t#a>0 = %d (of %d)" % (j, n_active[-1], len(_alpha)))
        print("Active dual variables: %d" % np.sum(n_active))

        sign_Delta_tj = np.sign(np.random.RandomState(10).randn(N, nE))

        # --- Using active set ---
        L_jbss_active = [L_jbss[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbtt_active = [L_jbtt[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbst_active = [L_jbst[j][:, alpha_j[j] > 0, :] for j in range(N)]
        L_jbts_active = [L_jbts[j][:, alpha_j[j] > 0, :] for j in range(N)]
        alpha_j_active = [alpha_j[j][alpha_j[j] > 0] for j in range(N)]

        S = np.repeat(np.column_stack(sign_Delta_tj), repeats=n_active, axis=1)
        M_jbss = np.concatenate(L_jbss_active, axis=1)
        M_jbtt = np.concatenate(L_jbtt_active, axis=1)
        M_jbst = np.concatenate(L_jbst_active, axis=1)
        M_jbts = np.concatenate(L_jbts_active, axis=1)
        Alphas = np.concatenate(alpha_j_active)

        start = time.time()
        expr = oe.contract_expression("ij,ijk,ijl,j", S, M_jbss.shape, M_jbtt.shape, Alphas, constants=[0, 3])
        A = expr(M_jbss, M_jbtt)
        B = expr(M_jbts, M_jbst)
        res_c = A - B
        print("C:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_c.shape)
        print(expr)

        start = time.time()
        res_d = np.zeros((nSs, nSt))
        for j in range(N):
            expr = oe.contract_expression("i,ijk,ijl,j",
                                          sign_Delta_tj[j],
                                          L_jbss_active[j].shape, L_jbtt_active[j].shape,
                                          alpha_j_active[j],
                                          constants=[0, 3])
            A = expr(L_jbss_active[j], L_jbtt_active[j])
            B = expr(L_jbts_active[j], L_jbst_active[j])
            res_d += (A - B)
        print("D:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_d.shape)
        print(expr)

        start = time.time()
        res_b = np.zeros((nSs, nSt))
        for j in range(N):
            A = oe.contract("i,ijk,ijl,j", sign_Delta_tj[j], L_jbss_active[j], L_jbtt_active[j], alpha_j_active[j])
            B = oe.contract("i,ijk,ijl,j", sign_Delta_tj[j], L_jbts_active[j], L_jbst_active[j], alpha_j_active[j])
            res_b += (A - B)
        print("B:\t%.5fs" % (time.time() - start))
        self.assertEqual((nSs, nSt), res_b.shape)

    def test_difference_vectors(self):
        n_edges = 50  # |E_j|
        n_active = 20  # |S_j|
        n_molecules = 100  # n_molecules

        n_rep = 50

        t_loop = 0.0
        t_einsum = 0.0

        for _ in range(n_rep):
            sign_delta_j = np.random.rand(n_edges)  # shape = (|E_j|, )
            A_j = np.random.rand(n_active)  # shape = (|S_j|, )
            L_delta = np.random.rand(n_edges, n_molecules, n_active)  # shape = (|E_j|, n_molecules, |S_j|)

            start = time.time()
            II_j_ref = np.zeros((n_molecules, ))
            for n in range(n_molecules):
                II_j_ref[n] = sign_delta_j @ L_delta[:, n, :] @ A_j
            t_loop += time.time() - start

            start = time.time()
            II_j = np.einsum("i,ikj,j", sign_delta_j, L_delta, A_j)
            t_einsum += time.time() - start

            np.testing.assert_allclose(II_j_ref, II_j)

        print("Loop: %.3fs" % (t_loop / n_rep))
        print("Einsum: %.3fs" % (t_einsum / n_rep))
