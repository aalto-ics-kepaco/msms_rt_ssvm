import unittest
import numpy as np

from ssvm.utils import item_2_idc


class Test_item_2_idc(unittest.TestCase):
    def test_correctness(self):
        l_Y = [[], ["A", "B", "C"], ["A"], ["D", "E"], ["D"], [], ["A", "X", "Z", "Y"], []]
        X = np.array([2, 2, 2, 3, 4, 4, 5, 7, 7, 7, 7])

        out = item_2_idc(l_Y)

        for s, Y in enumerate(l_Y):
            self.assertTrue(np.all(X[out[s]] == (s + 1)))
            self.assertEqual(len(Y), len(X[out[s]]))


if __name__ == '__main__':
    unittest.main()
