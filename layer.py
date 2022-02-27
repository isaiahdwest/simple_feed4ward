
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_prop(self, input):
        raise NotImplementedError
    

    def backward_prop(self, output_error, learning_rate):
        raise NotImplementedError

    def hello(self):
        print('hello!')

    
