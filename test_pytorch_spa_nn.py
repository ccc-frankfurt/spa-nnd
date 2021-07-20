import os
import unittest
import torch

import agrawal_example
import pytorch_spa_nn


class TestSpaNn(unittest.TestCase):
    def setUp(self):
        self.spa_nn = pytorch_spa_nn.SpaNn(network_architecture="spa")

        self._w_in_to_odd_agrawal = agrawal_example.w_in_to_odd
        self._w_odd_to_even_agrawal = agrawal_example.w_odd_to_even
        self._w_even_to_odd_agrawal = agrawal_example.w_even_to_odd
        self._w_even_to_out_agrawal = agrawal_example.w_even_to_out

        self._path_to_trained_fnn_for_activation_even_clipping = os.path.join(
            "test_data", "2021-06-22_10-08-07_ex_all_zero_5200cw_bs26_0_0"
        )
        self._checkpoint_trained_fnn_for_activation_even_clipping = torch.load(
            self._path_to_trained_fnn_for_activation_even_clipping)


    def test_activation_odd(self):
        test_results = self.spa_nn.activation_odd(
            even_layer_output=torch.tensor([[-1.4], [1.4], [-1.4], [1.4],[ -0.8], [0.8], [-0.8], [0.8], [-1], [0.2], [1], [2]], requires_grad=True),
            channel_input_LLR=torch.tensor([[-1], [0.5], [0.9], [-0.5], [1], [0.3], [1]]),  # random values
            w_in_to_odd=self._w_in_to_odd_agrawal,
            w_even_to_odd=self._w_even_to_odd_agrawal)

        # X_i = tanh ( 1/2 ( W^T_i20  x  L + W^T_e2o x X_i-1))
        # where W^T_i20  x  L = [-1, 0.9, -0.5, 1, -1, 0.5, 0.9, 0.3, 0.5, 0.9, -0.5, 1]
        # and W^T_e2o x X_i-1 = [-0.8, -0.6, 1, 0, -1.4, -1, 1.6, 0, 0.8, 0.6, -1.4, 0]
        # adding both lists up and multiplying by 1/2 results in:
        #[-0.9, 0.15, 0.25, 0.5, -1.2, -0.25, 1.25, 0.15, 0.65, 0.75, -0.95, 0.5]
        # to get the expected results tanh is used on this result
        # expected results are rounded to 3 decimal places
        expected_results = [-0.716, 0.149, 0.245, 0.462, -0.834, -0.245, 0.848, 0.149, 0.572, 0.635, -0.74, 0.462]
        rounded_test_results = (test_results * 10**3).round() / 10**3
        for i in range(len(expected_results)):
            self.assertEqual(rounded_test_results[i], expected_results[i])

    def test_activation_even(self):
        # odd layer output names are (O_{1,1}, O_{3,1}, O_{4,1}, O_{5,1}, O_{1,2}, O_{2,2}, O_{3,2}, O_{6,2}, O_{2,3}, O_{3,3}, O_{4,3}, O_{7,3})
        odd_layer_output = torch.tensor([[-0.716], [0.149], [0.245], [0.462], [-0.834], [-0.245], [0.848], [0.149], [0.572], [0.635], [-0.74], [0.462]])
        test_results = self.spa_nn.activation_even(
            odd_layer_output=odd_layer_output,
            w_odd_to_even=self._w_odd_to_even_agrawal
        )
        # expected result is
        # (E_{1,1}, E_{1,3}, E_{1,4}, E_{1,5}, E_{2,1}, E_{2,2}, E_{2,3}, E_{2,6}, E_{3,2}, E_{3,3}, E_{3,4}, E_{3,7}
        expected_results = [0.034, -0.162, -0.099, -0.052, -0.062, -0.212, 0.061, 0.35, -0.441, -0.396, 0.339, -0.551]
        rounded_test_results = (test_results * 10 ** 3).round() / 10 ** 3
        for i in range(len(expected_results)):
            self.assertEqual(rounded_test_results[i], expected_results[i])

    def test_activation_even_clipping_before_applying_tanh(self):
        spa_nn = pytorch_spa_nn.SpaNn(network_architecture="fnn")
        spa_nn.load_state_dict(self._checkpoint_trained_fnn_for_activation_even_clipping['model_state_dict'])


        odd_layer_output = torch.tensor([
            [-1.00000000],
            [-1.00000000],
            [-1.00000000],
            [-0.99999988],
            [-1.00000000],
            [ 1.00000000],
            [-1.00000000],
            [ 1.00000000],
            [ 1.00000000],
            [-1.00000000],
            [-1.00000000],
            [ 1.00000000]
        ])

        test_results = spa_nn.activation_even(
            odd_layer_output=odd_layer_output,
            w_odd_to_even=spa_nn.weights_even2  # weights_odd2 should also work if rnn is used
        )
        rounded_test_results = (test_results * 10 ** 4).round() / 10 ** 4
        expected_results = torch.tensor([
            [-16.6355],
            [-16.6355],
            [-16.6355],
            [-16.6355],
            [-16.6355],
            [ 16.6355],
            [-16.6355],
            [ 16.6355],
            [ 16.6355],
            [-16.6355],
            [-16.6355],
            [ 16.6355]])

        for i in range(len(expected_results)):
            self.assertEqual(rounded_test_results[i], expected_results[i])

    def test_activation_output(self):
        even_layer_output = torch.tensor([[-1.4], [1.4], [-1.4], [1.4], [-0.8], [0.8], [-0.8], [0.8], [-1], [0.2], [1], [2]])
        channel_input_LLR = torch.tensor([[-1], [0.5], [0.9], [-0.5], [1], [0.3], [1]])
        test_results = self.spa_nn.activation_output(
            even_layer_output=even_layer_output,
            w_even_to_out=self._w_even_to_out_agrawal,
            channel_input_LLR=channel_input_LLR
        )
        expected_results = [-3.2, 0.3, 1.7, -0.9, 2.4, 1.1, 3.0]  # rounded to 3 decimal places
        rounded_test_results = (test_results * 10 ** 3).round() / 10 ** 3
        for i in range(len(expected_results)):
            self.assertEqual(rounded_test_results[i], expected_results[i])

        # check if the dimensions of the output are ok, it should be a [7, 1] vector
        self.assertEqual(test_results.size()[0], 7)
        self.assertEqual(test_results.size()[1], 1)

    def test_cross_entropy_loss(self):
        # input is torch.sigmoid([-3.2, 0.3, 1.7, -0.9, 2.4, 1.1, 3.0])
        # = [0.03916572, 0.57444251, 0.84553480, 0.28905049, 0.91682732, 0.75026011, 0.95257413]
        prediction = torch.tensor([[0.039], [0.574], [0.846], [0.289], [0.917], [0.750], [0.953]])
        # target is hard decision on channel_input_LLR = [[-1], [0.5], [0.9], [-0.5], [1], [0.3], [1]]
        # where a positive number is mapped to 0 and a negative number is mapped to 1
        target = torch.tensor([[1], [0], [0], [1], [0], [0], [0]])
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 0.218
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_0bit_prediction_90_percent_correct(self):
        prediction = torch.tensor([[0.9]])  # we are 90% sure that this bit should be a 0
        target = torch.tensor([[0]])  # we expect that the bit is 0
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 0.105  # rounded result from -log(0.9)
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_1bit_prediction_90_percent_correct(self):
        prediction = torch.tensor([[0.1]])  # we are 10% sure that this bit is 0 so ith should be a 1
        target = torch.tensor([[1]])  # we expect that the bit is 1
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 0.105  # rounded result from -log(0.9)
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_0bit_prediction_100_percent_correct(self):
        prediction = torch.tensor([[1.]])  # we are 100% sure that this bit should be a 0
        target = torch.tensor([[0]])  # we expect that the bit is 0
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 0.  # our prediction is correct, we expect 0 error
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_1bit_prediction_100_percent_correct(self):
        prediction = torch.tensor([[0.]])  # there is a 0% chance that this bit should be a 0, therefore it must be a 1
        target = torch.tensor([[1]])  # we expect that the bit is 1
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 0. # our prediction is correct, we expect 0 error
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_bit_prediction_wrong_one(self):
        prediction = torch.tensor([[0.9]])  # we are 90% sure that this bit should be a 0
        target = torch.tensor([[1]])  # we expect that the bit is 1
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 2.303
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_bit_prediction_completely_wrong_zero(self):
        prediction = torch.tensor([[0.]])  # we are 0% sure that this bit should be a 0 -> 100% that bit should be 1
        target = torch.tensor([[0]])  # we expect that the bit is 0
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 36.044  # rounded result from -log(epsilon)
        self.assertEqual(rounded_test_result, expected_result)

    def test_cross_entropy_loss_one_bit_prediction_completely_wrong_one(self):
        prediction = torch.tensor([[1.]])  # we are 100% sure that this bit should be a 0
        target = torch.tensor([[1]])  # we expect that the bit is 1
        test_result = self.spa_nn.cross_entropy_loss(prediction, target)
        rounded_test_result = (test_result * 10 ** 3).round() / 10**3
        expected_result = 36.044  # rounded result from -log(epsilon)
        self.assertEqual(rounded_test_result, expected_result)


if __name__ == '__main__':
    unittest.main()
