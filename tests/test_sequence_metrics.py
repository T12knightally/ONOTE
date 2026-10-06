import unittest

from metrics.sequence_metrics import matching_block_count, matching_precision, sequence_similarity


class SequenceMetricTests(unittest.TestCase):
    def test_identical_sequences(self):
        values = ["C4", "D4", "E4"]
        self.assertEqual(matching_block_count(values, values), 3)
        self.assertEqual(sequence_similarity(values, values), 1.0)
        self.assertEqual(matching_precision(values, values), 1.0)

    def test_sequence_similarity_and_prediction_precision_differ(self):
        reference = ["C4", "D4", "E4", "F4"]
        prediction = ["C4", "E4", "F4"]
        self.assertEqual(matching_block_count(reference, prediction), 3)
        self.assertAlmostEqual(sequence_similarity(reference, prediction), 6 / 7)
        self.assertEqual(matching_precision(reference, prediction), 1.0)

    def test_empty_sequences_score_zero(self):
        self.assertEqual(sequence_similarity([], []), 0.0)
        self.assertEqual(matching_precision(["C4"], []), 0.0)
        self.assertEqual(matching_precision([], ["C4"]), 0.0)


if __name__ == "__main__":
    unittest.main()
