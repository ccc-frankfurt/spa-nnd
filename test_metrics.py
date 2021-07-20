import numpy as np
import unittest
import torch

import data
import metrics
import pytorch_spa_nn


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.untrained_spa_nn = pytorch_spa_nn.SpaNn(network_architecture="spa")
        self.codewords_in_training_dataset = 130
        self.batch_size = 13
        self.training_dataset = data.DataSet(
                batch_size=self.batch_size,
                number_of_codewords=self.codewords_in_training_dataset
        )
        self.train_input_llr, self.train_target = self.training_dataset.generate_data_set(
                codewords_per_snr_in_batch=self.training_dataset.codewords_per_snr_in_batch
        )
        self.x_train = torch.from_numpy(self.train_input_llr)
        self.y_train = torch.from_numpy(self.train_target).type(torch.int64)

    def test_accuracy_nn_one_codeword_with_errors(self):
        expected_bit_error_rate = 1/7
        expected_block_error_rate = 1

        bit_error_rate, block_error_rate = metrics.accuracy_nn(
            spa_nn=self.untrained_spa_nn,
            input_llr=torch.tensor(
                [[7.99],
                 [1.98],
                 [2.20],
                 [0.73],
                 [5.95],
                 [-2.57],  # this bit is going to be mapped to 1 instead of 0
                 [8.32]]),
            target=torch.zeros(7, 1).type(torch.int64)
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_nn_one_codeword_no_errors(self):
        expected_bit_error_rate = 0
        expected_block_error_rate = 0

        bit_error_rate, block_error_rate = metrics.accuracy_nn(
            spa_nn=self.untrained_spa_nn,
            input_llr=torch.tensor(
                [[7.99],
                 [1.98],
                 [2.20],
                 [0.73],
                 [5.95],
                 [2.57],
                 [8.32]]),
            target=torch.zeros(7, 1).type(torch.int64)
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_nn_two_codewords_both_have_errors(self):
        expected_bit_error_rate = 3 / 14
        expected_block_error_rate = 1

        input_llr = torch.tensor(
            [[7.99485636, -5.26927519],
             [2.20093536, 0.73450792],
             [-1.29319894, -2.57242894],
             [8.31911564, -5.67155027],
             [-2.76385689, 2.95138383],
             [0.31293672, -9.16788483],
             [-4.49046183, -4.65637064]])

        target = torch.tensor(
            [[1., 1.],
             [0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 0.],
             [1., 1.],
             [1., 1.]])

        bit_error_rate, block_error_rate = metrics.accuracy_nn(
            spa_nn=self.untrained_spa_nn,
            input_llr=input_llr,
            target=target.type(torch.int64)
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_nn_two_codewords_one_has_errors(self):
        expected_bit_error_rate = 2/14
        expected_block_error_rate = 1/2

        input_llr = torch.tensor(
            [[7.99485636, -5.26927519],
             [2.20093536, 0.73450792],
             [-1.29319894, -2.57242894],
             [8.31911564, -5.67155027],
             [-2.76385689, 2.95138383],
             [0.31293672, -9.16788483],
             [-4.49046183, -4.65637064]])

        target = torch.tensor(  # the first codeword is decoded properly, the second one has 2 errors
            [[0., 1.],
             [0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 0.],
             [1., 1.],
             [1., 1.]])

        bit_error_rate, block_error_rate = metrics.accuracy_nn(
            spa_nn=self.untrained_spa_nn,
            input_llr=input_llr,
            target=target.type(torch.int64)
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_classic_spa_perfect_decoding_one_codeword(self):
        # epsilon of 0.5/numbers_of_bits to replace a zero from perfect decoding bc NVS devides through SPAs BER
        epsilon = 0.5 / 7
        expected_bit_error_rate = epsilon
        expected_block_error_rate = 0

        bit_error_rate, block_error_rate = metrics.accuracy_classic_spa(
            input_llr=np.array([7.99, 1.98, 2.20, 0.73, 5.95, -2.57, 8.32]).reshape(1, 7),
            target=np.zeros(shape=(1, 7))
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_classic_spa_perfect_decoding_two_codewords(self):
        # epsilon of 0.5/numbers_of_bits to replace a zero from perfect decoding bc NVS devides through SPAs BER
        epsilon = 0.5 / 14
        expected_bit_error_rate = epsilon
        expected_block_error_rate = 0

        bit_error_rate, block_error_rate = metrics.accuracy_classic_spa(
            input_llr=np.array([[7.99, 1.98, 2.20, 0.73, 5.95, -2.57, 8.32],
                                [7.99, 1.98, 2.20, 0.73, 5.95,  2.57, 8.32]]),
            target=np.zeros(shape=(2, 7))
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_accuracy_classic_spa_one_codeword_with_erros(self):
        expected_bit_error_rate = 3/7
        expected_block_error_rate = 1

        bit_error_rate, block_error_rate = metrics.accuracy_classic_spa(
            input_llr=np.array([7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49]).reshape(1, 7),
            target=np.array([0, 0, 1, 0, 1, 1, 1]).reshape(1, 7)
        )
        self.assertEqual(expected_bit_error_rate, bit_error_rate)
        self.assertEqual(expected_block_error_rate, block_error_rate)

    def test_normalized_validation_score_one_snrs(self):
        bit_error_rate_nn = np.array([0.5])
        bit_error_rate_spa = np.array([0.5])

        nvs = metrics.calculate_normalized_validation_score(bit_error_rate_nn, bit_error_rate_spa)
        self.assertEqual(nvs, 1.0)

    def test_normalized_validation_score_two_snrs(self):
        bit_error_rate_nn = np.array([0.5, 0.7])  # BER for two different SNRs
        bit_error_rate_spa = np.array([0.5, 0.6])

        nvs = metrics.calculate_normalized_validation_score(bit_error_rate_nn, bit_error_rate_spa)
        self.assertEqual(round(nvs, 3), 1.083)

    def test_bers_per_snr_classic_spa_perfect_decoding_one_batch_two_SNRs(self):
        input_llr = np.array([[7.99, 1.98, 2.20, 0.73, 5.95, -2.57, 8.32],  # first SNR bucket in the batch
                              [7.99, 1.98, 2.20, 0.73, 5.95, 2.57, 8.32]])  # second SNR bucket in the batch
        target = np.zeros(shape=(2, 7))
        codewords_per_snr = 1
        batch_size = 2

        bers_per_snr = metrics.bers_per_snr_classic_spa(input_llr, target, codewords_per_snr, batch_size)

        bits_in_one_snr_bucket = codewords_per_snr * input_llr.shape[1]
        epsilon = 0.5 / bits_in_one_snr_bucket
        self.assertEqual(
            np.allclose(bers_per_snr, np.array([epsilon, epsilon])),
            True
        )

    def test_bers_per_snr_classic_spa_with_errors_two_batches_one_SNR(self):
        input_llr = np.array([
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, has bit_error_rate = 3/7
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, has bit_error_rate = 3/7
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 2, has bit_error_rate = 3/7
            [7.99, 1.98, 2.20, 0.73, 5.95, 2.57, 8.32]]   # Batch 2 has bit_error_rate = 0
        )                                                 # the SNR in all batches is the same
        target = np.array([
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 2
            [0, 0, 0, 0, 0, 0, 0]]  # Batch 2
        )
        codewords_per_snr = 2
        batch_size = 2
        bers_per_snr = metrics.bers_per_snr_classic_spa(input_llr, target, codewords_per_snr, batch_size)
        self.assertEqual(
            np.allclose(bers_per_snr, np.array([(3+3+3)/(4*7)])),
            True
        )

    def test_bers_per_snr_classic_spa_with_errors_two_batches_two_SNRs(self):
        input_llr = np.array([
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, SNR1, has bit_error_rate = 3/7
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, SNR1, has bit_error_rate = 3/7
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, SNR2, has bit_error_rate = 3/7
            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 1, SNR2, has bit_error_rate = 3/7

            [7.99, 2.20, -1.29, 8.32, -2.76, 8., -4.49],  # Batch 2, SNR1, has bit_error_rate = 3/7
            [7.99, 1.98, 2.20, 0.73, 5.95, 2.57, 8.32],   # Batch 2  SNR1, has bit_error_rate = 0
            [7.99, 1.98, 2.20, 0.73, 5.95, 2.57, 8.32],   # Batch 2  SNR2, has bit_error_rate = 0  Batch2 SNR2 has BER 0 -> use epsilon of 0.5
            [7.99, 1.98, 2.20, 0.73, 5.95, 2.57, 8.32]]   # Batch 2  SNR2, has bit_error_rate = 0
        )
        target = np.array([
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 1
            [0, 0, 1, 0, 1, 1, 1],  # Batch 2
            [0, 0, 0, 0, 0, 0, 0],  # Batch 2
            [0, 0, 0, 0, 0, 0, 0],  # Batch 2
            [0, 0, 0, 0, 0, 0, 0]]  # Batch 2
        )
        codewords_per_snr = 2
        batch_size = 4
        bers_per_snr = metrics.bers_per_snr_classic_spa(input_llr, target, codewords_per_snr, batch_size)
        # number_of_batches = 2
        # epsilon = 0.5 / (number_of_batches * codewords_per_snr * input_llr.shape[1])
        self.assertEqual(
            np.allclose(bers_per_snr, np.array([(3+3+3)/(4*7), (3+3+0.5)/(4*7)])),  # the 0.5 is the epsilon value
            True
        )

    def test_bers_per_snr_nn_with_errors_in_decoding_one_batch_one_snr_bucket(self):
        input_llr = torch.tensor(
            [[7.99485636, -5.26927519],
             [2.20093536, 0.73450792],
             [-1.29319894, -2.57242894],
             [8.31911564, -5.67155027],
             [-2.76385689, 2.95138383],
             [0.31293672, -9.16788483],
             [-4.49046183, -4.65637064]])

        target = torch.tensor(  # the first codeword is decoded properly, the second one has 2 errors
            [[0., 1.],
             [0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 0.],
             [1., 1.],
             [1., 1.]])

        codewords_per_snr = 2
        batch_size = 2
        bers_per_snr = metrics.bers_per_snr_nn(
            self.untrained_spa_nn,
            input_llr,
            target.type(torch.int64),
            codewords_per_snr,
            batch_size)

        self.assertEqual(
            np.allclose(bers_per_snr, np.array([2/14])),
            True
        )

    def test_bers_per_snr_nn_with_errors_in_decoding_one_batch_two_snr_bucket(self):
        input_llr = torch.tensor(
            [[7.99485636, -5.26927519],
             [2.20093536, 0.73450792],
             [-1.29319894, -2.57242894],
             [8.31911564, -5.67155027],
             [-2.76385689, 2.95138383],
             [0.31293672, -9.16788483],
             [-4.49046183, -4.65637064]])

        target = torch.tensor(  # the first codeword of SNR1 is decoded properly, the second one of SNR2 has 2 errors
            [[0., 1.],
             [0., 0.],
             [1., 0.],
             [0., 1.],
             [1., 0.],
             [1., 1.],
             [1., 1.]])

        codewords_per_snr = 1
        batch_size = 2
        bers_per_snr = metrics.bers_per_snr_nn(
            self.untrained_spa_nn,
            input_llr,
            target.type(torch.int64),
            codewords_per_snr,
            batch_size)

        self.assertEqual(
            # first codeword of SNR1 has 0 errors, second codeword of SNR2 has 2 erros
            np.allclose(bers_per_snr, np.array([0, 2/7])),
            True
        )

    def test_bers_per_snr_nn_with_errors_in_decoding_two_batches_two_snr_buckets(self):
        input_llr = torch.tensor(
            [[ 7.99485636,  7.99485636,  7.99485636, -5.26927519,   7.99485636,  7.99485636, -5.26927519, -5.26927519],
             [ 2.20093536,  2.20093536,  2.20093536,  0.73450792,   2.20093536,  2.20093536,  0.73450792,  0.73450792],
             [-1.29319894, -1.29319894, -1.29319894, -2.57242894,  -1.29319894, -1.29319894, -2.57242894, -2.57242894],
             [ 8.31911564,  8.31911564,  8.31911564, -5.67155027,   8.31911564,  8.31911564, -5.67155027, -5.67155027],
             [-2.76385689, -2.76385689, -2.76385689,  2.95138383,  -2.76385689, -2.76385689,  2.95138383,  2.95138383],
             [ 0.31293672,  0.31293672,  0.31293672, -9.16788483,   0.31293672,  0.31293672, -9.16788483, -9.16788483],
             [-4.49046183, -4.49046183, -4.49046183, -4.65637064,  -4.49046183, -4.49046183, -4.65637064, -4.65637064]])
             # Batch1 SNR1  Batch1 SNR1  Batch1 SNR2 Batch1 SNR2    Batch2 SNR1  Batch2 SNR1  Batch2 SNR2  Batch2 SNR2

        target = torch.tensor(                    # the first 3 codewords are decoded properly, the last has 2 errors
            [[0., 0., 0., 1.,   0., 0., 1., 1.],  # the first 2 codewords are in snr bucket one resulting in BER=0
             [0., 0., 0., 0.,   0., 0., 0., 0.],  # the second 2 codewors are in snr bucket two
             [1., 1., 1., 0.,   1., 1., 0., 0.],
             [0., 0., 0., 1.,   0., 0., 1., 1.],
             [1., 1., 1., 0.,   1., 1., 0., 0.],
             [1., 1., 1., 1.,   1., 1., 1., 1.],
             [1., 1., 1., 1.,   1., 1., 1., 1.]])
              # Batch 1            Batch 2

        codewords_per_snr = 2
        batch_size = 4

        bers_per_snr = metrics.bers_per_snr_nn(
            self.untrained_spa_nn,
            input_llr,
            target.type(torch.int64),
            codewords_per_snr,
            batch_size)

        self.assertEqual(
            np.allclose(bers_per_snr, np.array([0, 6/(4*7)])),  # 4*7 = overall codewords per SNR * codewordlength
            True
        )

    def test_compare_ber_and_ber_per_snr_with_only_one_snr_classic_spa(self):
        """
        If only one SNR is used in the dataset, the BER calculation should match the BER per SNR calculation.
        With classic SPA it is possible that the results do not match perfectly because of epsilon
        """
        ber_classic_spa_train, _ = metrics.accuracy_classic_spa(
            np.transpose(self.train_input_llr),
            np.transpose(self.train_target)
        )

        ber_classic_per_snr_spa_train = metrics.bers_per_snr_classic_spa(
            input_llr=np.transpose(self.train_input_llr),
            target=np.transpose(self.train_target),
            codewords_per_snr_in_batch=self.training_dataset.codewords_per_snr_in_batch,
            batch_size=13)

        self.assertAlmostEqual(ber_classic_spa_train, ber_classic_per_snr_spa_train[0][0], places=2)


    def test_compare_ber_and_ber_per_snr_with_only_one_snr_nn(self):
        """
        As there are no epsilons used within the BER generated by the neural network, the results
        have to match perfectly.
        """
        ber_spa_nn_train, _ = metrics.accuracy_nn(
            self.untrained_spa_nn,
            self.x_train,
            self.y_train
        )

        ber_per_snr_nn_train = metrics.bers_per_snr_nn(
            spa_nn=self.untrained_spa_nn,
            input_llr=self.x_train,
            target=self.y_train,
            codewords_per_snr_in_batch=self.training_dataset.codewords_per_snr_in_batch,
            batch_size=13)

        self.assertEqual(ber_spa_nn_train, ber_per_snr_nn_train[0][0])

    def test_calculate_simple_ber_comparision(self):
        bit_error_rate_nn = np.array([0.5, 0.7])  # BER for two different SNRs
        bit_error_rate_spa = np.array([0.5, 0.6])

        sbc = metrics.calculate_simple_ber_comparision(bit_error_rate_nn, bit_error_rate_spa)
        self.assertEqual(round(sbc, 3), 0.05)


if __name__ == '__main__':
    unittest.main()
