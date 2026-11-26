import unittest

from src.analysis import _paired_sign_flip_p


class PairedSignFlipTests(unittest.TestCase):
    def test_counts_ties_across_different_starting_rates(self):
        # Changes of +1, -1 and +1 in ten responses give the smallest possible absolute sum.
        totals = [10, 10, 10]
        p = _paired_sign_flip_p([3, 2, 4], totals, [4, 1, 5], totals, n_perm=127, seed=31)
        self.assertEqual(p, 1.0)

    def test_unequal_denominators_and_an_unobserved_condition(self):
        # Paired changes are -1/4, +1/4 and +1/2. With this seed, 94 draws meet the threshold.
        p = _paired_sign_flip_p([1, 0, 0, 0], [2, 3, 2, 0],
                                [1, 1, 1, 2], [4, 4, 2, 3], n_perm=127, seed=31)
        self.assertEqual(p, (94 + 1) / (127 + 1))
