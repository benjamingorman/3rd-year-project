import unittest

from .context import som2cmm
import som2cmm.cmm as cmm

class TestCMM(unittest.TestCase):

    def test_vec_to_str(self):
        vec = cmm.create_vector(3)
        vec[0,0] = 1
        vec[1,0] = 0

        vec[2,0] = 1
        self.assertEqual(cmm.binary_vec_to_str(vec), "101")

    def test_mat_to_str(self):
        mat = cmm.create_matrix(3, 3)
        mat[0,0] = 1
        mat[1,1] = 1
        mat[2,2] = 1
        expected = "100\n010\n001"
        self.assertEqual(cmm.binary_mat_to_str(mat), expected)
