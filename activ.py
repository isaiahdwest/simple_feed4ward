from layer import Layer

class ActivationLayer(Layer, activation, activation_prime):
    def __init__(self):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_prop(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_prop(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
