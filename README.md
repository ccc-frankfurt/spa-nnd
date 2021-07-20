# Sum-Product Algorithm Neural Network Decoder (SPA-NND)

This implementation of the Sum-Product Algorithm Neural Network Decoder (SPA-NND) was developed 
in the scope of the bachelor thesis _"On the Correspondence Between
Classic Coding Theory and Machine Learning"_. 
The Sum-Product Algorithm (SPA) is used for decoding error correcting codes like for 
example Low-Density Parity-Check Codes (LDPC codes). SPA is a message passing algorithm where 
the parity check matrix H is transformed into a bipartite graph called Tanner graph.

![Alt text](tanner-graph-from-h.png?raw=true "Tanner graph form parity-check matrix H")

Messages are passed along the edges from variable nodes to check nodes and back in each iteration. 
SPA is an approximate algorithm if the graph is not a tree. 

An approach of transforming the Sum-Product Algorithm into a neural network was first presented 
by Nachmani et al. in _"Deep Learning Methods for Improved Decoding of Linear
Codes"_ (see https://arxiv.org/abs/1706.07043) and treated in detail by Navneet Agrawal in _"Machine Intelligence in Decoding of Forward Error Correction
Codes"_ (see http://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A1146212&dswid=8240) in 2017. To create
a neural network the graph is unfolded for each iteration and the edges become nodes in the hidden layers. 

![Alt text](spa-nnd.png?raw=true "Neural Network Decoder for the example parity check matrix.")

## SPA-NND usage instructions
To start training adapt the parameters of the `train` function in `pytorch_train_network.py`,
there is no commandline parser implemented. The parameters are documented in the `train` 
function, the most important ones are highlighted here.
- `max_epochs`: last training epoch, normally in the range of [300, ..., 800] for the example used in the thesis
- `codewords_in_dataset_train`: number of codewords that the neural network will train on each epoch, pick from a range [512, ... 4096]
- `snr_range_train`: a np.array with signal-to-noise ratio (SNR) values in dB between [-2, ..., 4], it is a valid approach to pick only one SNR, try out 2 or 3 for a start. Negative SNRs alone will not lead to good results in training.  
- `snr_range_validation`: should be set to the same range as the snr_range_train to check for overfitting, can
                                 also be set to np.arange(-5, 8.5, 0.5) to match the realistic setup used in the test
                                 dataset.
- `dataset_state_train`: choose between "fixed" and "otf" (on the fly generated), otf will generate a new
                                training dataset each epoch, this will slow down training but produce a well trained
                                network after 100-200 epochs. The training loss will fluctuate a lot, which is expected
                                behaviour, if you want to check if the network is converging you need to take the
                                validation dataset loss as reference.
  
## Evaluation
The results for all experiments conducted in the thesis can be checked in `evaluation.ipynb`. 
Use plot functions from `plots.py` or the `evaluation.ipynb` Jupyter notebook (which also uses the `plots.py`
utility) to check on your network during training or evaluate the results when training has finished.
After the last epoch the loss and normalized-validation score (NVS) are plotted. 
Please note that the testset used in `evaluation.ipynb` has 54000 codewords. Therefore the execution of some cells may take 2-3 minutes. 
You can speedup the process by using a smaller `snr_test_range` with less SNRs in it. It is not 
recommended to use less than 2000 `codewords_per_snr_test`. When not enough codewords are used per SNR
the evaluation results depend heavily on the noise seed used and do not properly represent the performance of the 
trained network. 