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

from ssvm.ssvm import DualVariables, StructuredSVM


class TestStructuredSVM(unittest.TestCase):
    def test_is_feasible(self):
        N = 3
        C = 2

        # ------------------------------------
        # Check a feasible alpha
        alphas = [{("M1", "M4", "M9"): C / N},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (2 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (3 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertTrue(StructuredSVM._is_feasible(alphas, C))
        self.assertFalse(StructuredSVM._is_feasible(alphas, 1.9))
        self.assertFalse(StructuredSVM._is_feasible(alphas, 2.1))

        # ------------------------------------
        # Check a infeasible alpha
        alphas = [{("M1", "M4", "M9"): C / N},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (3 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (2 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertFalse(StructuredSVM._is_feasible(alphas, C))

        alphas = [{("M1", "M4", "M9"): - 1},
                  {("M5", "M1", "M10"): C / (2 * N), ("M6", "M2", "M1"): C / (3 * N)},
                  {("M2", "M1", "M3"): C / (3 * N), ("M5", "M2", "M1"): C / (2 * N), ("M10", "M1", "M7"): C / (3 * N)}]
        assert len(alphas) == N

        self.assertFalse(StructuredSVM._is_feasible(alphas, C))


class TestDualVariables(unittest.TestCase):
    def test_initialization(self):
        # ------------------------------------
        cand_ids = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]

        for C, N in [(1, 10), (0.5, 2), (2, 1)]:
            for n in range(1, 11):
                alphas = DualVariables(C=C, N=N, cand_ids=cand_ids, rs=11, num_init_active_vars=n)
                self.assertEqual(n, len(alphas))
                for key, a in alphas.items():
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(C / (N * n), a)

        # ------------------------------------
        cand_ids = [["M1", "M2", "M3", "M10", "M20"]]

        for C, N in [(1, 10), (0.5, 2), (2, 1)]:
            for n in range(1, 3):
                alphas = DualVariables(C=C, N=N, cand_ids=cand_ids, rs=11, num_init_active_vars=n)
                self.assertEqual(n, len(alphas))
                for key, a in alphas.items():
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(C / (N * n), a)

        # ------------------------------------
        cand_ids = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]
        alphas = [DualVariables(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=2),
                  DualVariables(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=1),
                  DualVariables(C=1.5, N=3, cand_ids=cand_ids, rs=12, num_init_active_vars=3)]
        self.assertTrue(StructuredSVM._is_feasible(alphas, 1.5))

    def test_update(self):
        cand_ids_1 = [["M1"], ["M1", "M2", "M3", "M10", "M20"], ["M4", "M5"], ["M2", "M4"], ["M7", "M9", "M8"]]

        # ------------------------------------
        gamma = 0.75
        C = 2
        N = 1
        alphas = DualVariables(C=C, N=N, cand_ids=cand_ids_1, rs=123, num_init_active_vars=2)

        # ('M1', 'M3', 'M4', 'M2', 'M8') and ('M1', 'M2', 'M5', 'M2', 'M9') are active.

        # Update an active dual variable
        y_seq_active = ('M1', 'M3', 'M4', 'M2', 'M8')
        a_old = alphas[y_seq_active]
        alphas.update(y_seq_active, gamma=gamma)
        self.assertTrue(alphas[y_seq_active] >= 0)
        self.assertTrue(alphas[y_seq_active] <= (C / N))
        self.assertEqual((1 - gamma) * a_old + gamma * (C / N), alphas[y_seq_active])

        # Update an inactive dual variable
        y_seq_inactive = ('M1', 'M20', 'M5', 'M2', 'M7')
        a_old = alphas[y_seq_inactive]
        self.assertEqual(0, a_old)
        alphas.update(y_seq_inactive, gamma=gamma)
        self.assertTrue(alphas[y_seq_inactive] >= 0)
        self.assertTrue(alphas[y_seq_inactive] <= (C / N))
        self.assertEqual((1 - gamma) * a_old + gamma * (C / N), alphas[y_seq_inactive])

        # ------------------------------------
        cand_ids_2 = [["M1", "M2"], ["M1", "M2", "M19", "M10"], ["M20", "M4", "M5"], ["M2"], ["M3", "M56", "M8"]]
        cand_ids_3 = [["M1", "M2", "M3"], ["M1", "M2", "M9", "M10", "M22"], ["M20", "M7"], ["M2", "M99"], ["M72", "M8"]]

        C = 2
        N = 3
        alphas = [DualVariables(C=C, N=N, cand_ids=cand_ids_1, rs=123, num_init_active_vars=2),
                  DualVariables(C=C, N=N, cand_ids=cand_ids_2, rs=121, num_init_active_vars=1),
                  DualVariables(C=C, N=N, cand_ids=cand_ids_3, rs=321, num_init_active_vars=4)]

        self.assertTrue(StructuredSVM._is_feasible(alphas, C))
        print(alphas[0].items())

        # Dual variable set belonging to example 0
        y_seq_active = ('M1', 'M3', 'M4', 'M2', 'M8')
        y_seq_inactive = ('M1', 'M20', 'M5', 'M2', 'M7')
        alphas[0].update(y_seq_active, gamma)
        print(alphas[0].items())
        self.assertTrue(StructuredSVM._is_feasible(alphas, C))
        alphas[0].update(y_seq_inactive, gamma)
        self.assertTrue(StructuredSVM._is_feasible(alphas, C))


if __name__ == '__main__':
    unittest.main()
