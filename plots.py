import itertools
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

import metrics


def plot_from_checkpoint_losses(checkpoint, last_plotted_epoch=None, ylim=(0, 0.3)):
    losses_train = checkpoint["losses_train"]
    codewords_in_dataset_train = checkpoint["codewords_in_dataset_train"]
    normalized_y_axis_losses_train = [loss / codewords_in_dataset_train for loss in losses_train]
    if last_plotted_epoch:
        normalized_y_axis_losses_train = normalized_y_axis_losses_train[:last_plotted_epoch+1]

    losses_validation = checkpoint["losses_validation"]
    x_axis_losses_validation = []
    y_axis_losses_validation = []
    for key, value in losses_validation.items():
        if last_plotted_epoch and key > last_plotted_epoch:
            break
        x_axis_losses_validation.append(key)
        y_axis_losses_validation.append(value)
    normalized_y_axis_losses_validation = np.array(y_axis_losses_validation) / checkpoint["codewords_in_dataset_validation"]

    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    x_axis = [i for i in range(checkpoint["epoch"] + 1)]
    if last_plotted_epoch:
        x_axis = [i for i in range(last_plotted_epoch + 1)]
    plt.plot(x_axis, normalized_y_axis_losses_train, label="Training Loss", color="green")
    plt.plot(x_axis_losses_validation, normalized_y_axis_losses_validation, label="Validation Loss", color="red")
    plt.ylim(ylim)
    plt.legend()
    batchsize = checkpoint["batch_size_train"]
    snr_range_train = checkpoint["snr_range_train"]
    architecture = checkpoint["network_architecture"]
    dataset_state_train = checkpoint["dataset_state_train"]
    plt.suptitle("Losses", fontsize=18, y=1.1)
    plt.title(f"{codewords_in_dataset_train} CW/E, BS {batchsize}, \ntrain SNR {snr_range_train}, \n{architecture.upper()} architecture, {dataset_state_train} dataset", fontsize=10)
    #plt.savefig(os.path.join("../", "bachelor_thesis", "plots", "experiments-fnn-vs-rnn", f"loss_{codewords_in_dataset_train}cw_{batchsize}bs_snr{snr_range_train}_{architecture}_{dataset_state_train}.pdf"), bbox_inches='tight')
    plt.show()


def plot_from_checkpoint_nvs(checkpoint, last_plotted_epoch=None, ylim=(0, 1)):
    dataset_state_train = checkpoint["dataset_state_train"]

    nn_BERs_per_snr_train = checkpoint["nn_BERs_per_snr_train"]
    spa_BER_per_snr_train = checkpoint["spa_BER_per_snr_train"]  # used for fixed training dataset
    spa_BERs_per_snr_train = checkpoint["spa_BERs_per_snr_train"]  # used for otf training dataset

    epochs = checkpoint["epoch"] + 1
    y_axis_nvs_train = []
    for epoch in range(epochs):
        if dataset_state_train == "fixed":
            nvs_tmp_train = metrics.calculate_normalized_validation_score(
                bit_error_rate_nn=np.array([nn_BERs_per_snr_train[epoch]]),
                bit_error_rate_spa=np.array([spa_BER_per_snr_train])
            )
        elif dataset_state_train == "otf":
            nvs_tmp_train = metrics.calculate_normalized_validation_score(
                bit_error_rate_nn=np.array([nn_BERs_per_snr_train[epoch]]),
                bit_error_rate_spa=np.array([spa_BERs_per_snr_train[epoch]])
            )
        y_axis_nvs_train.append(nvs_tmp_train)
    if last_plotted_epoch:
        y_axis_nvs_train = y_axis_nvs_train[:last_plotted_epoch+1]

    nn_BERs_per_snr_validation = checkpoint["nn_BERs_per_snr_validation"]
    spa_BER_per_snr_validation = checkpoint["spa_BER_per_snr_validation"]

    x_axis_nvs_validation = []
    y_axis_nvs_validation = []
    for epoch, value in nn_BERs_per_snr_validation.items():
        if last_plotted_epoch and epoch > last_plotted_epoch:
            break
        nvs_tmp_validation = metrics.calculate_normalized_validation_score(
            bit_error_rate_nn=np.array([value]),
            bit_error_rate_spa=np.array(spa_BER_per_snr_validation)
        )
        x_axis_nvs_validation.append(epoch)
        y_axis_nvs_validation.append(nvs_tmp_validation)

    plt.xlabel("Number of epochs")
    plt.ylabel("Normalized Validation Score")
    x_axis = [i for i in range(epochs)]
    if last_plotted_epoch:
        x_axis = [i for i in range(last_plotted_epoch + 1)]
    plt.plot(x_axis, y_axis_nvs_train, label="Training NVS", color="green")
    plt.plot(x_axis_nvs_validation, y_axis_nvs_validation, label="Validation NVS", color="red")
    codewords_in_dataset_train = checkpoint["codewords_in_dataset_train"]
    batchsize = checkpoint["batch_size_train"]
    architecture = checkpoint["network_architecture"]
    snr_range_train = checkpoint["snr_range_train"]
    plt.suptitle("Normalized Validation Scores", fontsize=18, y=1.1)
    plt.title(f"{codewords_in_dataset_train} CW/E, BS {batchsize}, \ntrain SNR {snr_range_train}, \n{architecture.upper()} architecture, {dataset_state_train} dataset", fontsize=10)
    plt.ylim(ylim)
    plt.legend()
    #plt.savefig(os.path.join("../", "bachelor_thesis", "plots", "experiments-fnn-vs-rnn", f"nvs_{codewords_in_dataset_train}cw_{batchsize}bs_snr{snr_range_train}_{architecture}_{dataset_state_train}.pdf"), bbox_inches='tight')
    plt.show()


def plot_from_checkpoint_average_ber(checkpoint, last_plotted_epoch=None, ylim=(0, 0.5)):
    nn_BERs_per_snr_train = checkpoint["nn_BERs_per_snr_train"]
    nn_BERs_summed = [np.sum(epoch) for epoch in nn_BERs_per_snr_train]

    snr_range_train = checkpoint["snr_range_train"]
    number_of_snrs = snr_range_train.size
    nn_BERs_summed_normalized = [summed_bers / number_of_snrs for summed_bers in nn_BERs_summed]
    if last_plotted_epoch:
        nn_BERs_summed_normalized = nn_BERs_summed_normalized[:last_plotted_epoch+1]

    nn_BERs_per_snr_validation = checkpoint["nn_BERs_per_snr_validation"]
    x_axis_validation = []
    y_axis_validation = []
    for epoch, value in nn_BERs_per_snr_validation.items():
        if last_plotted_epoch and epoch > last_plotted_epoch:
            break
        x_axis_validation.append(epoch)
        nn_BERs_summed_validation = np.sum(value)
        snr_range_validation = checkpoint["snr_range_validation"]
        number_of_snrs_validation = snr_range_validation.size
        nn_BERs_summed_normalized_validation = nn_BERs_summed_validation / number_of_snrs_validation
        y_axis_validation.append(nn_BERs_summed_normalized_validation)

    plt.xlabel("Number of epochs")
    plt.ylabel("Bit Error Rates")
    epochs = checkpoint["epoch"] + 1
    x_axis = [i for i in range(epochs)]
    if last_plotted_epoch:
        x_axis = [i for i in range(last_plotted_epoch + 1)]
    plt.plot(x_axis, nn_BERs_summed_normalized, label="Normalized training BER", color="green")
    plt.plot(x_axis_validation, y_axis_validation, label="Validation BER", color="red")
    codewords_in_dataset_train = checkpoint["codewords_in_dataset_train"]
    batchsize = checkpoint["batch_size_train"]
    architecture = checkpoint["network_architecture"]
    dataset_state_train = checkpoint["dataset_state_train"]
    plt.title(f"BER averaged over SNRs ({codewords_in_dataset_train} CW/E, BS {batchsize}, train SNR {snr_range_train}, {architecture} architecture, {dataset_state_train} dataset)")
    plt.ylim(ylim)
    plt.legend()
    plt.show()

def plot_from_checkpoint_compare_ber_between_snr(checkpoint, snr_train, snr_validation, ylim=(0, 0.5)):
    snr_range_train = checkpoint["snr_range_train"]
    index_of_snr_train = np.where(snr_range_train == snr_train)
    nn_BERs_per_snr_train = checkpoint["nn_BERs_per_snr_train"]
    nn_BERs_of_SNR = [float(epoch[index_of_snr_train]) for epoch in nn_BERs_per_snr_train]

    snr_range_validation = checkpoint["snr_range_validation"]
    index_of_snr_validation = np.where(snr_range_validation == snr_validation)
    nn_BERs_per_snr_validation = checkpoint["nn_BERs_per_snr_validation"]
    x_axis_validation = []
    y_axis_validation = []
    for epoch, value in nn_BERs_per_snr_validation.items():
        x_axis_validation.append(epoch)
        nn_BERs_of_SNR_validation = value[0][index_of_snr_validation]
        y_axis_validation.append(nn_BERs_of_SNR_validation)

    plt.xlabel("Number of epochs")
    plt.ylabel("Bit Error Rates")
    epochs = checkpoint["epoch"] + 1
    x_axis = [i for i in range(epochs)]
    plt.plot(x_axis, nn_BERs_of_SNR, label=f"BER training set SNR {snr_train}", color="green")
    plt.plot(x_axis_validation, y_axis_validation, label=f"BER validation set SNR {snr_validation}", color="red")
    plt.ylim(ylim)
    codewords_in_dataset_train = checkpoint["codewords_in_dataset_train"]
    batchsize = checkpoint["batch_size_train"]
    plt.title(f"Comparing BER for specific SNRs ({codewords_in_dataset_train} CW/E, BS {batchsize})")
    plt.legend()
    plt.show()


def plot_compare_spa_ber_and_nn_ber_test(checkpoint, spa_BER_per_snr_test, nn_BER_per_snr_test, snr_range_test, ylim=(0, 0.3)):
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.plot(snr_range_test, spa_BER_per_snr_test[0], linewidth=1, label=f"BER SPA")
    plt.plot(snr_range_test, nn_BER_per_snr_test[0], linewidth=1, label=f"BER trained Neural Network")
    #plt.yscale("log")  #  log scale not working as 0 values -> -inf
    plt.xticks(np.arange(snr_range_test[0], snr_range_test[-1]+1, 1.0))
    plt.ylim(ylim)
    plt.legend()
    codewords_in_dataset_train = checkpoint["codewords_in_dataset_train"]
    batchsize = checkpoint["batch_size_train"]
    snr_range_train = checkpoint["snr_range_train"]
    architecture = checkpoint["network_architecture"]
    dataset_state_train = checkpoint["dataset_state_train"]
    plt.suptitle("Comparing BERs using test dataset", fontsize=18, y=1.1)
    plt.title(f"{codewords_in_dataset_train} CW/E, BS {batchsize}, \ntrain SNR {snr_range_train}, \n{architecture.upper()} architecture, {dataset_state_train} dataset", fontsize=10)
    #plt.savefig(os.path.join("../", "bachelor_thesis", "plots", "experiments-fnn-vs-rnn", f"compare_ber_with_testset_{codewords_in_dataset_train}cw_{batchsize}bs_snr{snr_range_train}_{architecture}_{dataset_state_train}.pdf"), bbox_inches='tight')
    plt.show()


def plot_multiple_bers_per_snr(bers_to_plot, snr_range, title=f"Comparing testset BER of SPA and trained NN",
                               spa_BER_per_snr_test=None, ylim=(0.00001, 0.3), figsize=(14, 4.5)):
    plt.figure(0, figsize=figsize)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")

    if spa_BER_per_snr_test is not None:
        plt.plot(snr_range, spa_BER_per_snr_test[0], linewidth=1, label=f"BER SPA", marker="|")

    marker = itertools.cycle(('1', '2', '3', '4', '+', '.', '*'))
    for label, bers in bers_to_plot.items():
        plt.plot(snr_range, bers[0][0], linewidth=1, label=f"BER Neural Network ({label})", marker=next(marker))

    plt.yscale("log")
    plt.xticks(np.arange(snr_range[0], snr_range[-1] + 1, 1.0))
    plt.ylim(ylim)
    plt.legend()
    plt.title(title)
    #plt.savefig(os.path.join("../", "bachelor_thesis", "plots", "experiments-otf", "compare_bers_otf_batch_size.pdf"), bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    CHECKPOINT_PATH = os.path.join("checkpoints", "2021-07-02_11-10-09_ex_fnn_vs_rnn_f256")
    checkpoint = torch.load(CHECKPOINT_PATH)

    plot_from_checkpoint_losses(checkpoint)
    plot_from_checkpoint_nvs(checkpoint)


