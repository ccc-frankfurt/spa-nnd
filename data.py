import numpy as np
import pyldpc

from agrawal_example import parity_check_matrix_h


class DataSet:
    def __init__(
            self,
            parity_check_matrix=parity_check_matrix_h,
            generator_matrix=None,
            batch_size=1,
            number_of_codewords=1,
            use_all_zero_codeword_only=True,
            codeword_seed=0,
            noise_seed=1,
            snr=np.array([2])):

        self.parity_check_matrix = parity_check_matrix
        self.generator_matrix = generator_matrix
        self.batch_size = batch_size
        self.number_of_codewords = number_of_codewords
        self.use_all_zero_codeword_only = use_all_zero_codeword_only
        self.codeword_seed = codeword_seed
        self.noise_seed = noise_seed
        self.snr = snr  # list of SNRs in db we want to train on

        self.codeword_random = np.random.RandomState(self.codeword_seed)
        self.noise_random = np.random.RandomState(self.noise_seed)

        # calculate generator matrix if it is not specified
        if self.generator_matrix is None:
            self.generator_matrix = pyldpc.code.coding_matrix(self.parity_check_matrix)

        if self.number_of_codewords % self.batch_size == 0:
            self.number_of_batches = self.number_of_codewords // self.batch_size
        else:
            raise ValueError(f"number_of_codewords {self.number_of_batches} must be dividable by batch_size {self.batch_size}.")

        if self.batch_size % len(self.snr) == 0:
            self.codewords_per_snr_in_batch = self.batch_size // len(self.snr)
        else:
            raise ValueError(f"batch_size {self.batch_size} must be dividable by the number of given SNR's {self.snr}.")

        self.codeword_length = self.parity_check_matrix.shape[1]
        self.number_of_parity_checks = self.parity_check_matrix.shape[0]
        self.number_of_information_bits = self.codeword_length - self.number_of_parity_checks
        # the rate of the code defined by (number of information bits)/(transmitted bits) cf. Agrawal p. 11
        self.rate = self.number_of_information_bits / self.codeword_length
        self.variances = self.calculate_variances_from_snr(self.snr)  # variance aka scaling_factor

    def calculate_variances_from_snr(self, snr):
        # SNR cf. Agrawal p. 48 (3.31)
        variances = np.sqrt(1 / (2 * 10**(snr/10) * self.rate))
        return variances

    def generate_codewords(self, number_of_codewords):
        if self.use_all_zero_codeword_only:
            codewords = np.zeros((self.codeword_length, number_of_codewords))

        else:
            messages = self.codeword_random.randint(0, 2, size=(self.number_of_information_bits,
                                                                number_of_codewords))
            codewords = np.dot(self.generator_matrix, messages) % 2

        return codewords

    def generate_bpsk_codewords(self, codewords):
        # maps 1 -> -1 and maps 0 -> 1
        return (-1)**codewords

    def generate_noisy_bpsk_codewords(self, bpsk_codewords, variance):
        noise = self.noise_random.normal(loc=0,  # mean aka expected value
                                         scale=variance,
                                         size=bpsk_codewords.shape)
        return bpsk_codewords + noise

    def generate_LLR_of_noisy_bpsk_codewords(self, noisy_bpsk_codewords, variance):
        # cf. Agrawal p. 6 formula (2.1)
        LLR = (2 * noisy_bpsk_codewords) / (variance ** 2)
        LLR_clipped_upper_boundary = np.where(LLR > 10, 10, LLR)
        LLR_clipped_both_boundarys = np.where(LLR_clipped_upper_boundary < -10, -10, LLR_clipped_upper_boundary)
        return LLR_clipped_both_boundarys

    def generate_data_set(self, codewords_per_snr_in_batch):
        batches = [self.generate_batch(codewords_per_snr_in_batch) for i in range(self.number_of_batches)]
        input_llr = np.hstack([b[0] for b in batches])
        target = np.hstack([b[1] for b in batches])
        return input_llr.astype(np.float32), target.astype(np.float32)

    def generate_batch(self, number_of_codewords):
        batch = [self.generate_input_llr_for_variance(variance, number_of_codewords) for variance in self.variances]
        input_llr = np.hstack([b[0] for b in batch])
        target = np.hstack([b[1] for b in batch])
        return input_llr.astype(np.float32), target.astype(np.float32)

    def generate_input_llr_for_variance(self, variance, number_of_codewords):
        target = self.generate_codewords(number_of_codewords)
        bpsk_codewords = self.generate_bpsk_codewords(target)
        noisy_bpsk_codewords = self.generate_noisy_bpsk_codewords(bpsk_codewords, variance)
        input_llr = self.generate_LLR_of_noisy_bpsk_codewords(noisy_bpsk_codewords, variance)
        return input_llr, target

