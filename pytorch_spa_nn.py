import copy
import torch
import torch.nn as nn

import agrawal_example


class SpaNn(torch.nn.Module):
    def __init__(self, network_architecture):
        """
        :param network_architecture: "spa" for sum product algorithm where weights in network are not trained
                                     "fnn" for a feed forward structure of the neural network
                                     "rnn" for a recurrent neural network structure with shared weights in even layers
        """
        super(SpaNn, self).__init__()
        self.network_architecture = network_architecture

        # masks are not trainable, are bool because the ~ operator when updating weights for masking needs bool
        self.register_buffer("in_to_odd_mask", agrawal_example.w_in_to_odd.type(torch.bool))
        self.register_buffer("odd_to_even_mask", agrawal_example.w_odd_to_even.type(torch.bool))
        self.register_buffer("even_to_odd_mask", agrawal_example.w_even_to_odd.type(torch.bool))
        self.register_buffer("even_to_out_mask", agrawal_example.w_even_to_out.type(torch.bool))

        # fixed helper values for activation_odd in first layer because there is no input from previous layers yet
        self.register_buffer("first_even_layer_output", torch.zeros(self.in_to_odd_mask.shape[1], 1))
        self.register_buffer("first_w_even_to_odd", torch.zeros(self.even_to_odd_mask.shape))

        # fixed helper values for activation_even
        self.register_buffer(
            "matrix_of_ones",
            torch.ones(agrawal_example.w_even_to_odd.shape[0],
                       agrawal_example.w_even_to_odd.shape[0]).float())
        self.register_buffer("scalar_one", torch.tensor([1.], dtype=torch.float32))
        self.register_buffer("scalar_minus_one", torch.tensor([-1.], dtype=torch.float32))
        self.register_buffer("upper_boundary", torch.tensor(1. - torch.finfo(torch.float32).eps))
        self.register_buffer("lower_boundary", torch.tensor(-1. + torch.finfo(torch.float32).eps))

        # weights for one iteration
        self.in_to_odd = agrawal_example.w_in_to_odd.type(torch.float)
        self.odd_to_even = agrawal_example.w_odd_to_even.type(torch.float)
        self.even_to_odd = agrawal_example.w_even_to_odd.type(torch.float)
        self.even_to_out = agrawal_example.w_even_to_out.type(torch.float)

        if self.network_architecture == "spa":
            self.weights_odd1 = nn.Parameter(self.in_to_odd, requires_grad=False)
            self.weights_even1 = nn.Parameter(self.odd_to_even, requires_grad=False)
            self.weights_out = nn.Parameter(self.even_to_out, requires_grad=False)
            self.weights_odd2 = nn.Parameter(self.even_to_odd, requires_grad=False)

        elif self.network_architecture == "fnn":  # feed forward neural network
            # in_to_odd is the connection from input to first hidden layer, typically not trained cf. Agrawal p. 33
            self.weights_odd1 = nn.Parameter(self.in_to_odd.detach(), requires_grad=False)
            self.weights_even1 = nn.Parameter(self.odd_to_even.detach(), requires_grad=False)
            # eights_odd2 are the connections from the end of iteration 1 to start of iteration 2
            self.weights_odd2 = nn.Parameter(copy.deepcopy(self.even_to_odd.detach()))
            self.weights_even2 = nn.Parameter(self.odd_to_even.detach(), requires_grad=False)
            self.weights_odd3 = nn.Parameter(copy.deepcopy(self.even_to_odd.detach()))
            self.weights_even3 = nn.Parameter(self.odd_to_even.detach(), requires_grad=False)
            self.weights_odd4 = nn.Parameter(copy.deepcopy(self.even_to_odd.detach()))
            self.weights_even4 = nn.Parameter(self.odd_to_even.detach(), requires_grad=False)
            self.weights_odd5 = nn.Parameter(copy.deepcopy(self.even_to_odd.detach()))
            self.weights_even5 = nn.Parameter(self.odd_to_even.detach(), requires_grad=False)
            # weights_out are the connections to the output layer, typically not trained cf. Agrawal p. 33
            self.weights_out = nn.Parameter(self.even_to_out.detach(), requires_grad=False)

        elif self.network_architecture == "rnn": # recurrent neural network
            # all iterations share the same weight in the recurrent neural network decoder, cf. Agrawal. p. 39
            self.weights_odd1 = nn.Parameter(self.in_to_odd, requires_grad=False)
            self.weights_even1 = nn.Parameter(self.odd_to_even, requires_grad=False)
            self.weights_out = nn.Parameter(self.even_to_out, requires_grad=False)
            self.weights_odd2 = nn.Parameter(self.even_to_odd)

    def activation_odd(self, even_layer_output, channel_input_LLR, w_in_to_odd, w_even_to_odd):
        w_in_to_odd_transposed = torch.transpose(w_in_to_odd, 0, 1).float()  # pytorch can't multiply int and float
        w_even_to_odd_transposed = torch.transpose(w_even_to_odd, 0, 1).float()  # cast to float
        return torch.tanh(0.5 * (
                    torch.matmul(w_in_to_odd_transposed, channel_input_LLR) + torch.matmul(w_even_to_odd_transposed,
                                                                                           even_layer_output)))

    def activation_even(self, odd_layer_output, w_odd_to_even):
        repeated_odd_layer_output = odd_layer_output.repeat(1, w_odd_to_even.shape[0])

        # 2) calculate auxiliary matrix $M^{*}_{i-1} = W_{odd2even} \odot M_{i-1}$, \odot = element-wise multiplication
        auxilary_matrix = torch.mul(w_odd_to_even, repeated_odd_layer_output)

        # 3) Replace zeros in auxiliary matrix $M^{*}_{i-1}$ with ones
        auxilary_matrix_with_ones = torch.where((auxilary_matrix == 0), self.matrix_of_ones,
                                                auxilary_matrix.float())

        # 4) Auxiliary vector $X^{*}_{i-1}$ of size $|E|$ created by multiplying along column elements of $M^{*}_{i-1}$.
        # The products in $X^{*}_{i-1}$ correspond to
        # $\prod_{t \in adj(c_i) \setminus j} (1 - 2 \cdot \mathbb{P}^{(\text{in})}(w_t = 1))$.
        auxiliary_vector = torch.prod(auxilary_matrix_with_ones, dim=0)  # build product along every column
        auxiliary_vector = torch.unsqueeze(auxiliary_vector, 0)  # prod removes dimension, make it (1, number_of_edges)

        # 5) Calculate the output vector $X_i$ of the even layer $T_i$ using $tanh^{-1}$ as activation function.
        # clip input to 1-epsilon and (-1 + epsilon) to prevent tanh^{-1} from outputting infinity
        # clipping before entering the value into torch.atanh is necessary to prevent nans produced by pytorchs outograd
        clipped_upper_auxiliary_vector = torch.where(
            auxiliary_vector >= self.scalar_one,
            self.upper_boundary,
            auxiliary_vector
        )
        clipped_lower_auxiliary_vector = torch.where(
            clipped_upper_auxiliary_vector <= self.scalar_minus_one,
            self.lower_boundary,
            clipped_upper_auxiliary_vector
        )
        even_layer_output = 2 * torch.atanh(clipped_lower_auxiliary_vector)
        return torch.transpose(even_layer_output, 0, 1)

    def activation_output(self, even_layer_output, w_even_to_out, channel_input_LLR):
        # cf. Agrawal p. 28, equation 3.10
        # \overline{W_i} = R + W^{T}_{even2out} \otimes T_{i-1}
        w_even_to_out_transposed = torch.transpose(w_even_to_out, 0, 1).float()
        even_layer_output = torch.reshape(
            even_layer_output,
            (even_layer_output.shape[0], 1)
        )  # vertical vectors need a reshape otherwise broadcasting results in a wrong output dimension
        return channel_input_LLR + torch.matmul(w_even_to_out_transposed, even_layer_output)

    def cross_entropy_loss(self, prediction, target):
        # cf. Agrawal p. 42 (3.19)
        # when the prediction is 100% right or 100% wrong a log(0) = infinity is calculated even if this result is
        # multiplied by 0 afterwards, pytorch struggles with handling infinity as it produces nan values during
        # backpropagation. Fixing critical values by adding/subtracting epsilon so the smallest input to log function
        # is set to epsilon
        epsilon = torch.finfo(torch.double).eps
        prediction_fix_ones = torch.where(prediction == 1, (1-epsilon), prediction.type(torch.double))
        prediction_with_epsilons = torch.where(prediction == 0, epsilon, prediction_fix_ones.type(torch.double))

        codeword_length = target.shape[0]
        summands = target * torch.log(1 - prediction_with_epsilons) + (1 - target) * torch.log(prediction_with_epsilons)
        result = torch.sum(summands) / (-codeword_length)
        return result.type(torch.float32)  # cast back to float32 from where operation that needed double

    def forward_fnn(self, x):
        odd1_output = self.activation_odd(
            even_layer_output=self.first_even_layer_output,  # init with 0 bc no E_ij availabe yet
            channel_input_LLR=x,
            w_in_to_odd=self.weights_odd1,
            w_even_to_odd=self.first_w_even_to_odd)  # result of even_layer_output * w_even_to_odd is 0
        even1_output = self.activation_even(odd_layer_output=odd1_output, w_odd_to_even=self.weights_even1)

        odd2_output = self.activation_odd(
                even_layer_output=even1_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd2)
        even2_output = self.activation_even(odd_layer_output=odd2_output, w_odd_to_even=self.weights_even2)

        odd3_output = self.activation_odd(
                even_layer_output=even2_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd3)
        even3_output = self.activation_even(odd_layer_output=odd3_output, w_odd_to_even=self.weights_even3)

        odd4_output = self.activation_odd(
                even_layer_output=even3_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd4)
        even4_output = self.activation_even(odd_layer_output=odd4_output, w_odd_to_even=self.weights_even4)

        odd5_output = self.activation_odd(
                even_layer_output=even4_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd5)
        even5_output = self.activation_even(odd_layer_output=odd5_output, w_odd_to_even=self.weights_even5)


        output5_output = self.activation_output(
                even_layer_output=even5_output,
                w_even_to_out=self.weights_out,
                channel_input_LLR=x)
        return torch.sigmoid(output5_output)

    def forward_rnn(self, x):
        odd1_output = self.activation_odd(
            even_layer_output=self.first_even_layer_output,  # init with 0 bc no E_ij availabe yet
            channel_input_LLR=x,
            w_in_to_odd=self.weights_odd1,
            w_even_to_odd=self.first_w_even_to_odd)  # result of even_layer_output * w_even_to_odd is 0
        even1_output = self.activation_even(odd_layer_output=odd1_output, w_odd_to_even=self.weights_even1)

        odd2_output = self.activation_odd(
                even_layer_output=even1_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd2)
        even2_output = self.activation_even(odd_layer_output=odd2_output, w_odd_to_even=self.weights_even1)

        odd3_output = self.activation_odd(
                even_layer_output=even2_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd2)
        even3_output = self.activation_even(odd_layer_output=odd3_output, w_odd_to_even=self.weights_even1)

        odd4_output = self.activation_odd(
                even_layer_output=even3_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd2)
        even4_output = self.activation_even(odd_layer_output=odd4_output, w_odd_to_even=self.weights_even1)

        odd5_output = self.activation_odd(
                even_layer_output=even4_output,
                channel_input_LLR=x,
                w_in_to_odd=self.weights_odd1,
                w_even_to_odd=self.weights_odd2)
        even5_output = self.activation_even(odd_layer_output=odd5_output, w_odd_to_even=self.weights_even1)

        output5_output = self.activation_output(
                even_layer_output=even5_output,
                w_even_to_out=self.weights_out,
                channel_input_LLR=x)
        return torch.sigmoid(output5_output)

    def forward(self, x):
        if self.network_architecture == "fnn":
            return self.forward_fnn(x)

        elif self.network_architecture == "rnn" or self.network_architecture == "spa":
            return self.forward_rnn(x)
