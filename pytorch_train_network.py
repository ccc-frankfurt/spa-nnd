import torch
import numpy as np
import os
from datetime import datetime

import data
import metrics
import plots
from pytorch_spa_nn import SpaNn

torch.set_printoptions(linewidth=180, precision=8, edgeitems=12)
np.set_printoptions(linewidth=180)


def continue_training_from_checkpoint(checkpoint, max_epochs, new_checkpoint_path, device):
    spa_nn_continued = SpaNn(checkpoint["network_architecture"])
    spa_nn_continued.load_state_dict(checkpoint['model_state_dict'])
    spa_nn_continued_on_device = spa_nn_continued.to(device)

    optimizer_continued = torch.optim.RMSprop(
        params=spa_nn_continued_on_device.parameters(),
        lr=0.001,
        alpha=0.99,
        eps=1e-08,
        weight_decay=0,
        momentum=0,
        centered=False)
    optimizer_continued.load_state_dict(checkpoint['optimizer_state_dict'])

    train(spa_nn_continued_on_device,
          optimizer_continued,
          new_checkpoint_path,
          device=device,
          start_epoch=checkpoint["epoch"] + 1,
          max_epochs=max_epochs,
          codewords_in_dataset_train=checkpoint["codewords_in_dataset_train"],
          batch_size_train=checkpoint["batch_size_train"],
          snr_range_train=checkpoint["snr_range_train"],
          snr_range_validation=checkpoint["snr_range_validation"],
          codewords_per_snr_validation=checkpoint["codewords_per_snr_validation"],
          dataset_state_train=checkpoint["dataset_state_train"],
          interval_for_creating_checkpoint=checkpoint["interval_for_creating_checkpoint"],
          continue_from_checkpoint=True,
          checkpoint=checkpoint,
          use_all_zero_codeword_only_train=checkpoint["use_all_zero_codeword_only_train"],
          use_all_zero_codeword_only_validation=checkpoint["use_all_zero_codeword_only_validation"])


def train(spa_nn,
          optimizer,
          checkpoint_path,
          device="cpu",
          start_epoch=0,
          max_epochs=200,
          codewords_in_dataset_train=4,
          batch_size_train=2,
          snr_range_train=np.array([2]),
          snr_range_validation=np.array([2]),
          codewords_per_snr_validation=500,
          dataset_state_train="fixed",
          interval_for_creating_checkpoint=20,
          continue_from_checkpoint=False,
          checkpoint={},
          use_all_zero_codeword_only_train=True,
          use_all_zero_codeword_only_validation=True):
    """
    Trains the given initialized spa_nn.
    :param spa_nn: an initialized neuronal network from the SpaNn class, can be either fnn, rnn or untrainable spa
    :param optimizer: a torch.optim optimizer usually RMSProp
    :param checkpoint_path: path where regular checkpoints of the neural network training status are stored,
                            the checkpoint will also be used for plots and evaluation after training, checkpoints
                            can also be used to resume training from
    :param device: either cpu or gpu
    :param start_epoch: this is per default set to 0 but will be changed if training is resumed
    :param max_epochs: last training epoch, normally in the range of [300, ..., 800] for the example used in the thesis
    :param codewords_in_dataset_train: number of codewords that the neural network will train on each epoch
    :param batch_size_train: number of codewords after that an optimizer.step() is performed
    :param snr_range_train: a np.array with signal-to-noise ratio values in dB between [-2, ..., 4]
    :param snr_range_validation: should be set to the same range as the snr_range_train to check for overfitting, can
                                 also be set to np.arange(-5, 8.5, 0.5) to match the realistic setup used in the test
                                 dataset.
    :param codewords_per_snr_validation: using 500 codewords per snr in validation dataset provided robust results, be
                                         aware that your validation dataset size will be
                                         snr_range_validation.shape[0] * codewords_per_snr_validation,
                                         picking a wide snr range in the validation set will slow down training
    :param dataset_state_train: choose between "fixed" and "otf" (on the fly generated), otf will generate a new
                                training dataset each epoch, this will slow down training but produce a well trained
                                network after 100-200 epochs. The training loss will fluctuate a lot, which is expected
                                behaviour, if you want to check if the network is converging you need to take the
                                validation dataset loss as reference.
    :param interval_for_creating_checkpoint: determine after how many epoch you want to create a backup for the state
                                             of your network
    :param continue_from_checkpoint: set to True if training is to be continued, set False if you are starting anew
    :param checkpoint: this parameter is initialized with {}, it is only relevant if training is continued from a
                       checkpoint
    :param use_all_zero_codeword_only_train: set to True for using the all zero codeword + noise, set to false if you
                                             want to train on randomly generated codewords + noise
    :param use_all_zero_codeword_only_validation: set to True for using the all zero codeword + noise, set to false
                                                  for randomly generated codewords + noise
    """

    # generate datasets
    if dataset_state_train == "fixed":
        dataset_train = data.DataSet(batch_size=batch_size_train,
                                     number_of_codewords=codewords_in_dataset_train,
                                     use_all_zero_codeword_only=use_all_zero_codeword_only_train,
                                     snr=snr_range_train,
                                     noise_seed=11)
        input_llr_train, target_train = dataset_train.generate_data_set(
            codewords_per_snr_in_batch=dataset_train.codewords_per_snr_in_batch)

        x_train = torch.from_numpy(input_llr_train)
        x_train = x_train.to(device)
        y_train = torch.from_numpy(target_train).type(torch.int64)
        y_train = y_train.to(device)

        spa_BLER_per_snr_train = None  # todo
        spa_BER_per_snr_train = metrics.bers_per_snr_classic_spa(
            input_llr=np.transpose(input_llr_train),
            target=np.transpose(target_train),
            codewords_per_snr_in_batch=dataset_train.codewords_per_snr_in_batch,
            batch_size=batch_size_train)

    # validation dataset
    codewords_in_dataset_validation = snr_range_validation.size * codewords_per_snr_validation
    dataset_validation = data.DataSet(
        number_of_codewords=codewords_in_dataset_validation,
        batch_size=codewords_in_dataset_validation,
        use_all_zero_codeword_only=use_all_zero_codeword_only_validation,
        snr=snr_range_validation,
        codeword_seed=4,
        noise_seed=5,
    )
    input_llr_validation, target_validation = dataset_validation.generate_data_set(
        codewords_per_snr_in_batch=dataset_validation.codewords_per_snr_in_batch)
    x_validation = torch.from_numpy(input_llr_validation)
    x_validation = x_validation.to(device)
    y_validation = torch.from_numpy(target_validation).type(torch.int64)
    y_validation = y_validation.to(device)

    spa_BLER_per_snr_validation = None  # todo
    spa_BER_per_snr_validation = metrics.bers_per_snr_classic_spa(
        input_llr=np.transpose(input_llr_validation),
        target=np.transpose(target_validation),
        codewords_per_snr_in_batch=dataset_validation.codewords_per_snr_in_batch,
        batch_size=codewords_in_dataset_validation)

    if continue_from_checkpoint:
        losses_train = checkpoint["losses_train"]
        nn_BLERs_per_snr_train = checkpoint["nn_BLERs_per_snr_train"]
        nn_BERs_per_snr_train = checkpoint["nn_BERs_per_snr_train"]
        spa_BLERs_per_snr_train = checkpoint["spa_BLERs_per_snr_train"]
        spa_BERs_per_snr_train = checkpoint["spa_BERs_per_snr_train"]

        losses_validation = checkpoint["losses_validation"]
        nn_BLERs_per_snr_validation = checkpoint["nn_BLERs_per_snr_validation"]
        nn_BERs_per_snr_validation = checkpoint["nn_BERs_per_snr_validation"]
    else:
        losses_train = []
        nn_BLERs_per_snr_train = []
        nn_BERs_per_snr_train = []
        spa_BLERs_per_snr_train = []  # will only be filled with values if training dataset is otf
        spa_BERs_per_snr_train = []  # will only be filled with values if training dataset is otf

        losses_validation = {}
        nn_BLERs_per_snr_validation = {}
        nn_BERs_per_snr_validation = {}

    for epoch in range(start_epoch, max_epochs):
        loss_epoch = 0
        loss_batch = 0

        # generate on the fly training dataset for epoch
        if dataset_state_train == "otf":
            dataset_train = data.DataSet(batch_size=batch_size_train,
                                         number_of_codewords=codewords_in_dataset_train,
                                         use_all_zero_codeword_only=use_all_zero_codeword_only_train,
                                         snr=snr_range_train,
                                         noise_seed=epoch)  # use a different seed every epoch
            input_llr_train, target_train = dataset_train.generate_data_set(
                codewords_per_snr_in_batch=dataset_train.codewords_per_snr_in_batch)

            x_train = torch.from_numpy(input_llr_train)
            x_train = x_train.to(device)
            y_train = torch.from_numpy(target_train).type(torch.int64)
            y_train = y_train.to(device)

            spa_BLER_per_snr_train = None  # todo
            spa_BLERs_per_snr_train.append(spa_BLER_per_snr_train)
            spa_BER_per_snr_train = metrics.bers_per_snr_classic_spa(
                input_llr=np.transpose(input_llr_train),
                target=np.transpose(target_train),
                codewords_per_snr_in_batch=dataset_train.codewords_per_snr_in_batch,
                batch_size=batch_size_train)
            spa_BERs_per_snr_train.append(spa_BER_per_snr_train)

        for i in range(1, codewords_in_dataset_train + 1):
            x = x_train[:, i - 1:i]  # slices one column from the dataset
            y = y_train[:, i - 1:i]
            prediction = spa_nn(x)
            loss = spa_nn.cross_entropy_loss(prediction, y)
            loss_batch += loss.detach()  # for tracking

            if spa_nn.network_architecture != "spa":
                loss.backward()

            # update weights after one batch, apply mask then use optimizer to update
            if (i % batch_size_train) == 0:
                with torch.no_grad():
                    for name, parameter in spa_nn.named_parameters():
                        if "weights_odd1" in name and parameter.requires_grad:
                            # mask source: https://discuss.pytorch.org/t/backpropagate-with-respect-to-mask/35336/4)
                            parameter.grad[~spa_nn.in_to_odd_mask] = 0  # ~ to invert mask
                        elif "weights_odd" in name and parameter.requires_grad:
                            parameter.grad[~spa_nn.even_to_odd_mask] = 0
                        elif "weights_out" in name and parameter.requires_grad:
                            parameter.grad[~spa_nn.even_to_out_mask] = 0

                optimizer.step()
                optimizer.zero_grad()
                loss_epoch += loss_batch
                loss_batch = 0

        losses_train.append(loss_epoch)

        with torch.no_grad():
            # todo create a block error rate per snr calculation
            nn_BLER_per_snr_train = None
            nn_BLERs_per_snr_train.append(nn_BLER_per_snr_train)

            nn_BER_per_snr_train = metrics.bers_per_snr_nn(
                spa_nn=spa_nn,
                input_llr=x_train,
                target=y_train,
                codewords_per_snr_in_batch=dataset_train.codewords_per_snr_in_batch,
                batch_size=batch_size_train)
            nn_BERs_per_snr_train.append(nn_BER_per_snr_train)

            # validation how the model performs so far
            loss_validation = 0
            for i in range(1, codewords_in_dataset_validation + 1):
                x_t = x_validation[:, i - 1:i]  # slices one column from the dataset
                y_t = y_validation[:, i - 1:i]
                prediction_t = spa_nn(x_t)
                loss_t = spa_nn.cross_entropy_loss(prediction_t, y_t)
                loss_validation += loss_t.detach()
            losses_validation[epoch] = loss_validation

            nn_BLER_per_snr_validation = None  # todo
            nn_BLERs_per_snr_validation[epoch] = nn_BLER_per_snr_validation

            nn_BER_per_snr_validation = metrics.bers_per_snr_nn(
                spa_nn=spa_nn,
                input_llr=x_validation,
                target=y_validation,
                codewords_per_snr_in_batch=dataset_validation.codewords_per_snr_in_batch,
                batch_size=codewords_in_dataset_validation)
            nn_BERs_per_snr_validation[epoch] = nn_BER_per_snr_validation

        if epoch % interval_for_creating_checkpoint == 0 or epoch == (max_epochs - 1):  # always store the last epoch
            torch.save({
                "max_epochs": max_epochs,
                "epoch": epoch,
                "network_architecture": spa_nn.network_architecture,
                "model_state_dict": spa_nn.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "batch_size_train": batch_size_train,
                "dataset_state_train": dataset_state_train,
                "snr_range_train": snr_range_train,
                "snr_range_validation": snr_range_validation,
                "codewords_per_snr_validation": codewords_per_snr_validation,
                "codewords_in_dataset_train": codewords_in_dataset_train,
                "codewords_in_dataset_validation": codewords_in_dataset_validation,
                "use_all_zero_codeword_only_train": use_all_zero_codeword_only_train,
                "use_all_zero_codeword_only_validation": use_all_zero_codeword_only_validation,
                "losses_train": losses_train,
                "losses_validation": losses_validation,
                "nn_BERs_per_snr_train": nn_BERs_per_snr_train,
                "nn_BERs_per_snr_validation": nn_BERs_per_snr_validation,
                "nn_BLER_per_snr_train": nn_BLER_per_snr_train,
                "nn_BLERs_per_snr_validation": nn_BLERs_per_snr_validation,
                "nn_BLERs_per_snr_train": nn_BLERs_per_snr_train,
                "spa_BLERs_per_snr_train": spa_BLERs_per_snr_train,
                "spa_BER_per_snr_train": spa_BER_per_snr_train,
                "spa_BER_per_snr_validation": spa_BER_per_snr_validation,
                "spa_BLER_per_snr_train": spa_BLER_per_snr_train,
                "spa_BLER_per_snr_validation": spa_BLER_per_snr_validation,
                "spa_BLERs_per_snr_train": spa_BLERs_per_snr_train,  # will be an empty list if dataset train is fixed
                "spa_BERs_per_snr_train": spa_BERs_per_snr_train,  # will be an empty list if dataset train is fixed
                "interval_for_creating_checkpoint": interval_for_creating_checkpoint,
                "best_model": {epoch: spa_nn.state_dict()},  # todo determine best model according to nvs validation?
            }, checkpoint_path)

        print(f"{datetime.now()} Results from epoch {epoch} \n\
        training loss: {loss_epoch}, validation loss: {losses_validation[max(list(losses_validation.keys()))]}\n\
        training BER: {nn_BER_per_snr_train}, validation BER: {nn_BERs_per_snr_validation[max(list(nn_BERs_per_snr_validation.keys()))]}")
        print("---------------------NEXT EPOCH-----------------------------------------------------------------")

    checkpoint = torch.load(checkpoint_path)
    plots.plot_from_checkpoint_losses(checkpoint)
    plots.plot_from_checkpoint_nvs(checkpoint)


if __name__ == "__main__":
    device = "cpu"  #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    continue_training = False

    if continue_training:
        path_to_checkpoint_to_continue_from = os.path.join("checkpoints", "2021-07-08_09-57-28_ex_fnn_vs_rnn_r4096")
        checkpoint = torch.load(path_to_checkpoint_to_continue_from)
        continue_training_from_checkpoint(checkpoint, 800, path_to_checkpoint_to_continue_from, device)

    else:
        checkpoint_path = os.path.join(
            "checkpoints",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_test_cleanup_pytorch_spa_nn")
        spa_nn = SpaNn("fnn")
        spa_nn_on_device = spa_nn.to(device)
        optimizer = torch.optim.RMSprop(params=spa_nn_on_device.parameters(),
                                        lr=0.001,
                                        alpha=0.99,
                                        eps=1e-08,
                                        weight_decay=0,
                                        momentum=0,
                                        centered=False)
        train(spa_nn_on_device, optimizer, checkpoint_path, device, codewords_in_dataset_train=200, batch_size_train=20)


