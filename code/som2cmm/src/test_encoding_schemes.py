import unittest
import numpy as np
from . import encoding_schemes as enc
from . import cmm

def count_1s(code):
    return sum(code)

class TestQuantizationEncoder(unittest.TestCase):
    test_cases = [
        (1, 0.0,   [1,0,0,0,0], 0.0),
        (1, 0.199, [1,0,0,0,0], 0.0),
        (1, 0.2,   [0,1,0,0,0], 0.2),
        (1, 0.4,   [0,0,1,0,0], 0.4),
        (1, 0.99,  [0,0,0,0,1], 0.8),
        (1, 1.0,   [0,0,0,0,1], 0.8),
        (2, 0.0,   [1,1,0,0,0], 0.0),
        (2, 0.099, [1,1,0,0,0], 0.0),
        (2, 0.100, [1,0,1,0,0], 0.1),
        (2, 0.200, [1,0,0,1,0], 0.2),
        (2, 0.301, [1,0,0,0,1], 0.3),
        (2, 0.401, [0,1,1,0,0], 0.4),
        (2, 0.501, [0,1,0,1,0], 0.5),
        (2, 0.601, [0,1,0,0,1], 0.6),
        (2, 0.701, [0,0,1,1,0], 0.7),
        (2, 0.801, [0,0,1,0,1], 0.8),
        (2, 0.901, [0,0,0,1,1], 0.9),
        (2, 1.000, [0,0,0,1,1], 0.9),
        (3, 0.0,   [1,1,1,0,0], 0.0),
        (3, 0.099, [1,1,1,0,0], 0.0),
        (3, 0.101, [1,1,0,1,0], 0.1),
        (3, 0.201, [1,1,0,0,1], 0.2),
        (3, 0.301, [1,0,1,1,0], 0.3),
        (3, 0.401, [1,0,1,0,1], 0.4),
        (3, 0.501, [1,0,0,1,1], 0.5),
        (3, 0.601, [0,1,1,1,0], 0.6),
        (3, 0.701, [0,1,1,0,1], 0.7),
        (3, 0.801, [0,1,0,1,1], 0.8),
        (3, 0.901, [0,0,1,1,1], 0.9),
        (3, 1.000, [0,0,1,1,1], 0.9),
        ]

    def test_binomial(self):
        self.assertEqual(enc.binomial(5, 2), 10)

    def test_encode_attr(self):
        qe = enc.QuantizationEncoder([], [], [])

        vmin = 0
        vmax = 1
        bits_used = 5

        for (bits_set, attr, expected, _) in self.test_cases:
            code = qe.encode_attr(attr, vmin, vmax, bits_used, bits_set)
            print("bits set {0}, attr {1}".format(bits_set, attr))
            self.assertEqual(code, expected)

    def test_decode_attr(self):
        qe = enc.QuantizationEncoder([], [], [])

        vmin = 0
        vmax = 1
        bits_used = 5

        print("decoding")

        for (bits_set, original_attr, code, expected_attr) in self.test_cases:
            attr = qe.decode_attr(code, vmin, vmax, bits_used, bits_set)
            np.testing.assert_almost_equal(attr, expected_attr)

    def test_encode(self):
        qe = enc.QuantizationEncoder(
                [(0, 2), (0, 5), (0, 8)],
                [5, 5, 5],
                [1, 1, 1]
                )
        code = qe.encode([1.2, 4.5, 7.8])
        expected = [0,0,0,0,0,
                    0,0,0,0,0,
                    0,0,0,0,0,
                    ]
        self.assertEqual(len(code), sum(qe.bits_per_attr))
        self.assertEqual(count_1s(code), sum(qe.bits_set_per_attr))

    def test_decode(self):
        qe1 = enc.QuantizationEncoder(
                [(0, 10), (0, 5), (0, 2)],
                [10, 5, 2],
                [1, 1, 1]
                )

        qe2 = enc.QuantizationEncoder(
                [(0, 10), (0, 5), (0, 3)],
                [10, 5, 3],
                [1, 1, 1]
                )

        patterns = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 4.0, 0.0],
                [3.0, 4.0, 1.0],
                ]

        for pat in patterns:
            code1 = qe1.encode(pat)
            decode1 = qe1.decode(code1)

            code2 = qe2.encode(pat)
            decode2 = qe2.decode(code2)

            self.assertEqual(decode1, pat)
            self.assertEqual(decode2, pat)
