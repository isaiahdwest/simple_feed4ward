class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # adding layers
    def add(self, layer):
        self.layers.append(layer)

    # set loss
    def lossFunc(self, loss, loss_prime):

        self.loss = loss
        self.loss_prime = loss_prime

    # predict output
    def predict(self, input_data):
        # sample dims
        samples = len(input_data)
        result = []

        # run net over all samples
        for i in range(samples):
            # forward prop
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_prop(output)

            result.append(output)

        return result

    # fit the net
    def fit(self, x_train, y_train, epochs = 50, learning_rate = 0.1):
        # sample dims
        samples = len(x_train)

        # training loop:
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward prop
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                err += self.loss(y_train[j], output)

                # back prop
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, learning_rate)
            err /= samples
            print('epoch %d/%d error=%f' % (i+1, epochs, err))





            
