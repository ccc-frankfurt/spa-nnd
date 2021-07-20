import numpy as np
import torch

import agrawal_example
from channelcoding_ldpc import ldpc_decode


def hard_decision_LLR(output_LLR):
    return torch.where(output_LLR >= 0, 0, 1)  # positive values are mapped to 0, negative values are mapped to 1


def hard_decision_sigmoid(output_with_sigmoid_activation):
    """
    Sigmoid function is applied to the output layer to obtain probability of a bit being 0 from its LLR value.
    A value >= 0.5 from the sigmoid activation is therefore mapped to 0, values <0.5 mapped to 1.
    cf. Agrawal p. 29
    """
    return torch.where(output_with_sigmoid_activation >= 0.5, 0, 1)


def accuracy_nn(spa_nn, input_llr, target):
    dataset_size = target.shape[1]
    codeword_length = target.shape[0]

    bit_errors = 0  # wrongly decoded bits
    block_errors = 0  # wrongly decoded codewords
    for i in range(1, dataset_size + 1):
        x = input_llr[:, i - 1:i]  # slices one column from the dataset
        y = target[:, i - 1:i]

        with torch.no_grad():
            prediction = hard_decision_sigmoid(spa_nn(x))
            if not torch.equal(prediction, y):
                block_errors += 1
                for bit in range(codeword_length):
                    if y[bit] != prediction[bit]:
                        bit_errors += 1

    bit_error_rate = bit_errors / torch.numel(target)
    block_error_rate = block_errors / dataset_size

    return bit_error_rate, block_error_rate


def accuracy_classic_spa(input_llr, target):
    dataset_size = target.shape[0]
    codeword_length = target.shape[1]

    bit_errors = 0  # wrongly decoded bits
    block_errors = 0  # wrongly decoded codewords
    for i in range(dataset_size):
        codeword_in = input_llr[i]
        # his LLR maps 1 -> positive, 0 -> negative which is just the other way round then I do it
        # if he returns the all one codeword he correctly decoded the all zero codeword
        # for decoding multiply noisy codeword with -1 before using ldpc_decode for the correct result
        decoded_by_spa = ldpc_decode(agrawal_example.parity_check_matrix_h, codeword_in * -1, 5)
        target_iter = target[i]
        if not np.array_equal(target_iter, decoded_by_spa):
            block_errors += 1
            for bit in range(codeword_length):
                if target_iter[bit] != decoded_by_spa[bit]:
                    bit_errors += 1

    if bit_errors == 0:
        # replace a BER of zero with an epsilon as dividing through BER from classic SPA is needed for NVS
        # for epsilon a value smaller than the smallest possible BER of 1/target.size is used
        epsilon = 0.5 / target.size
        bit_error_rate = epsilon

    else:
        bit_error_rate = bit_errors / target.size

    block_error_rate = block_errors / dataset_size

    return bit_error_rate, block_error_rate


def calculate_normalized_validation_score(bit_error_rate_nn, bit_error_rate_spa):
    """
    The Normalized Validation Score (NVS) is used to determine how well the NND performs wrt to the SPA decoder
    cf. Agrawal p. 33
    """
    normalized_score = bit_error_rate_nn / bit_error_rate_spa
    summed_score = np.sum(normalized_score)
    averaged_score = summed_score / bit_error_rate_nn.shape[-1]
    return averaged_score


def evaluate_loss(spa_nn, input_llr, target):
    losses = 0
    dataset_size = target.shape[1]
    for i in range(1, dataset_size + 1):
        x = input_llr[:, i - 1:i]  # slices one column from the dataset
        y = target[:, i - 1:i]
        with torch.no_grad():
            prediction = spa_nn(x)
            loss = spa_nn.cross_entropy_loss(prediction, y)
            losses += loss
    return losses


def bers_per_snr_classic_spa(input_llr, target, codewords_per_snr_in_batch, batch_size):
    """
    Bit error rates from classic SPA per SNR do not change during training, they are a fixed value.
    Therefore it only needs to be calculated once.
    The inputs are the numpy arrays created by the data module.
    """
    codewords_in_dataset = target.shape[0]
    number_of_batches = codewords_in_dataset // batch_size
    number_of_snrs = int(batch_size / codewords_per_snr_in_batch)
    bers_per_snr = np.zeros((1, number_of_snrs))

    for i in range(0, codewords_in_dataset, batch_size):  # for each batch
        batch_codewords = input_llr[i:i + batch_size]
        batch_targets = target[i:i + batch_size]
        bers_in_batch = []
        for j in range(0, batch_size, codewords_per_snr_in_batch): # for each SNR bucket in the batch

            snr_codewords = batch_codewords[j:j + codewords_per_snr_in_batch]
            snr_targets = batch_targets[j:j + codewords_per_snr_in_batch]
            ber, _ = accuracy_classic_spa(input_llr=snr_codewords, target=snr_targets)

            # the ber is calculated wrt the number of bits in this batches snr bucket
            # but it needs to be calculated wrt the number of ALL bits in a SNR bucket not only one batch
            # BER = errand_bits / number_of_bits_in_batch_in_SNR_bucket
            # BER needs to be: errand_bits / number_of_bits_in_ALL_batches_in_SNR_bucket
            #                = errand_bits / (number_of_bits_in_batch_in_SNR_bucket * number_of_batches)
            #                ==> divide result by number_of_batches
            ber_normalized_over_all_batches = ber / number_of_batches
            bers_in_batch.append(ber_normalized_over_all_batches)

        bers_per_snr += np.array(bers_in_batch)

    return bers_per_snr


def bers_per_snr_nn(spa_nn, input_llr, target, codewords_per_snr_in_batch, batch_size):
    """
    Bit error rates from the neural network decoder change during training depending on the state of the neural network.
    Therefore they need to be calculated every epoch for training and test dataset.
    The input are torch tensors like the ones used within the neural network.

    target must be of type torch.int64
    """
    codewords_in_dataset = input_llr.shape[1]
    number_of_batches = codewords_in_dataset // batch_size
    number_of_snrs = int(batch_size / codewords_per_snr_in_batch)
    bers_per_snr = np.zeros((1, number_of_snrs))

    for i in range(0, codewords_in_dataset, batch_size):  # for each batch
        batch_input_llr = input_llr[:, i:i + batch_size]
        batch_target = target[:, i:i + batch_size]
        bers_in_batch = []
        for j in range(0, batch_size, codewords_per_snr_in_batch): # for each SNR bucket in the batch
            input_llr_snr = batch_input_llr[:, j:j + codewords_per_snr_in_batch]  # columns with same SNR
            target_snr = batch_target[:, j:j + codewords_per_snr_in_batch]
            ber, _ = accuracy_nn(spa_nn, input_llr_snr, target_snr)
            ber_normalized_over_all_batches = ber / number_of_batches
            bers_in_batch.append(ber_normalized_over_all_batches)
        bers_per_snr += np.array(bers_in_batch)
    return bers_per_snr


def calculate_simple_ber_comparision(bit_error_rate_nn, bit_error_rate_spa):
    """
    The Simple BER Comparison(SBC) is used to determine how well the NND performs wrt to the SPA decoder
    """
    normalized_score = bit_error_rate_nn - bit_error_rate_spa
    summed_score = np.sum(normalized_score)
    averaged_score = summed_score / bit_error_rate_nn.shape[-1]
    return averaged_score


def calculate_test_dataset_sbc_per_epoch(checkpoint):
    """
    The bit error rates from SPA are the same for each epoch.
    :param checkpoint: dictionary loaded with torch.load(<path_to_checkpoint_file>)
    """
    test_bit_error_rates_per_snr = checkpoint["test_bit_error_rates_per_snr"]
    ber_classic_per_snr_spa_test = checkpoint["ber_classic_per_snr_spa_test"]
    epochs = len(test_bit_error_rates_per_snr)

    sbc_test = []
    for e in range(epochs):
        sbc_test_temp = calculate_simple_ber_comparision(
            bit_error_rate_nn=np.array(test_bit_error_rates_per_snr[e]),
            bit_error_rate_spa=np.array(ber_classic_per_snr_spa_test))
        sbc_test.append(sbc_test_temp)

    return sbc_test


def calculate_validation_dataset_sbc_per_epoch(checkpoint):
    """
    :param checkpoint: dictionary loaded with torch.load(<path_to_checkpoint_file>)
    """

    validation_bit_error_rates_per_snr = checkpoint["validation_bit_error_rates_per_snr"]
    ber_classic_per_snr_spa_validation = checkpoint["ber_classic_per_snr_spa_validation"]

    sbc_validation = {}
    for epoch, value in validation_bit_error_rates_per_snr.items():
        sbc_validation_temp = calculate_simple_ber_comparision(
            bit_error_rate_nn=np.array(value),
            bit_error_rate_spa=np.array(ber_classic_per_snr_spa_validation)
        )
        sbc_validation[epoch] = sbc_validation_temp

    return sbc_validation


def evaluate_values_for_last_epoch(checkpoint):
    last_epoch_loss_train = checkpoint["losses_train"][-1]/checkpoint["codewords_in_dataset_train"]
    print(f"Last epoch loss train = {round(float(last_epoch_loss_train), 3)}")
    max_epoch_validation = max(checkpoint["losses_validation"].keys())
    last_epoch_loss_validation = checkpoint["losses_validation"][max_epoch_validation]/checkpoint["codewords_in_dataset_validation"]
    print(f"Last epoch loss validation = {round(float(last_epoch_loss_validation), 3)}")

    last_epoch_nn_BERs_per_snr_train = checkpoint["nn_BERs_per_snr_train"][-1]
    last_epoch_spa_BER_per_snr_train = checkpoint["spa_BER_per_snr_train"][-1]

    nvs_train = calculate_normalized_validation_score(
        bit_error_rate_nn=np.array(last_epoch_nn_BERs_per_snr_train),
        bit_error_rate_spa=np.array(last_epoch_spa_BER_per_snr_train))
    print(f"NVS train last epoch = {round(nvs_train, 3)}")

    last_epoch_nn_BERs_per_snr_validation = checkpoint["nn_BERs_per_snr_validation"][max_epoch_validation]
    last_epoch_spa_BER_per_snr_validation = checkpoint["spa_BER_per_snr_validation"]  # only one value for all epochs

    nvs_validation = calculate_normalized_validation_score(
        bit_error_rate_nn=np.array(last_epoch_nn_BERs_per_snr_validation),
        bit_error_rate_spa=np.array(last_epoch_spa_BER_per_snr_validation))
    print(f"NVS validation last epoch = {round(nvs_validation, 3)}")


