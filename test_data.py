import unittest
import numpy as np

import data


class TestDataSet(unittest.TestCase):
    def setUp(self):
        pass

    def test_calculate_variance_from_default_snr(self):
        test_data_set = data.DataSet()
        calculated_variance = test_data_set.variances
        self.assertEqual(np.round(calculated_variance, 2), [0.74])

    def test_calculate_variance_from_snr_positive_snr(self):
        test_data_set = data.DataSet()
        snr = np.array([4])  # cf. Agrawal p. 49 optimal choice of training SNR is in the range -2dB and 4dB
        calculated_variance = test_data_set.calculate_variances_from_snr(snr)
        self.assertEqual(np.round_(calculated_variance, 2).tolist(), [0.59])

    def test_calculate_variance_from_snr_multiple_snrs(self):
        test_data_set = data.DataSet(number_of_codewords=2, batch_size=2)
        snr = np.array([-2, 0])  # cf. Agrawal p. 49 the optimal choice of the training SNR value lies in the range
        # -2dB and 4dB
        calculated_variance = test_data_set.calculate_variances_from_snr(snr)
        self.assertEqual(np.round_(calculated_variance, 2).tolist(), [1.18, 0.94])

    def test_generate_codewords_all_zero_word(self):
        test_data_set = data.DataSet()
        codewords = test_data_set.generate_codewords(number_of_codewords=test_data_set.number_of_codewords)
        self.assertEqual(codewords.shape, (7, 1))
        self.assertEqual(np.all(codewords == 0), True)

    def test_generate_codewords_ones_allowed(self):
        test_data_set = data.DataSet(use_all_zero_codeword_only=False)
        codewords = test_data_set.generate_codewords(number_of_codewords=test_data_set.number_of_codewords)  # message is (0, 1, 1, 0)^T
        expected_codewords = np.array([[1], [0], [0], [0], [1], [1], [0]])
        self.assertEqual(codewords.shape, (7, 1))
        self.assertEqual(codewords.tolist(), expected_codewords.tolist())

    def test_generate_codewords_ones_allowed_two_codewords(self):
        test_data_set = data.DataSet(
            use_all_zero_codeword_only=False,
            batch_size=2,
            number_of_codewords=2)
        codewords = test_data_set.generate_codewords(number_of_codewords=test_data_set.number_of_codewords)  # message is (0, 1, 1, 0)^T
        expected_codewords = np.array([[0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
        self.assertEqual(codewords.shape, (7, 2))
        self.assertEqual(codewords.tolist(), expected_codewords.tolist())

    def test_generate_batch_shapes(self):
        test_data_set = data.DataSet(
            batch_size=2,
            number_of_codewords=2,
            snr=np.array([2, 3]))
        input_llr, target = test_data_set.generate_batch(number_of_codewords=test_data_set.codewords_per_snr_in_batch)
        self.assertEqual(input_llr.shape, (7, 2))
        self.assertEqual(target.shape, (7, 2))

    def test_generate_batch_values(self):
        test_data_set = data.DataSet()
        input_llr, target = test_data_set.generate_batch(number_of_codewords=test_data_set.codewords_per_snr_in_batch)
        expected_input_llr = np.array([[7.99],
                              [1.98],
                              [2.20],
                              [0.73],
                              [5.95],
                              [-2.57],
                              [8.32]], dtype=np.float32)
        rounded_input_llr = np.around(input_llr, 2)
        self.assertEqual(True, np.array_equal(rounded_input_llr, expected_input_llr))
        self.assertEqual(input_llr.shape, (7, 1))
        self.assertEqual(target.shape, (7, 1))

    def test_generate_data_set_shapes_one_batch(self):
        test_data_set = data.DataSet()
        expected_input_llr, expected_target = test_data_set.generate_data_set(
            codewords_per_snr_in_batch=test_data_set.codewords_per_snr_in_batch)
        self.assertEqual(expected_input_llr.shape, (7, 1))
        self.assertEqual(expected_target.shape, (7, 1))

    def test_generate_data_set_shapes_multiple_batches(self):
        test_data_set = data.DataSet(
            batch_size=2,
            number_of_codewords=4,
            snr=np.array([2, 3]))
        expected_input_llr, expected_target = test_data_set.generate_data_set(
            codewords_per_snr_in_batch=test_data_set.codewords_per_snr_in_batch)
        self.assertEqual(expected_input_llr.shape, (7, 4))
        self.assertEqual(expected_target.shape, (7, 4))





