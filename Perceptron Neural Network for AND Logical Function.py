import numpy as np

# region NeuralNetwork
class NeuralNetwork():

    def __init__(self, learning_rate, threshold):
        self.learning_rate = learning_rate
        self.threshold = threshold
        np.random.seed(1)  # generate random array
        self.synaptic_weights = 2 * np.random.random((2, 1)) - 1  # generate 2 random weights values

    # compare the output of summation with the threshold
    def step(self, x):
        if x > float(self.threshold):
            return 1
        else:
            return 0
        pass

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            self.synaptic_weights += np.dot(training_inputs.T, error * self.learning_rate)
        pass

    def think(self, inputs):
        inp0uts = inputs.astype(float)
        output = self.step(np.sum(np.dot(inputs, self.synaptic_weights)))
        return output
        pass


# endregion
# region Neural_Network_Main_Fn
def NN_Main():
    learning_rate = 0.1
    threshold = -0.2
    neural_network = NeuralNetwork(learning_rate, threshold)

    print("Beginning Randomly Generated Weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0],
                                [0, 1],
                                [1, 0],
                                [1, 1]])

    training_outputs = np.array([[0, 0, 0, 1]]).T

    neural_network.train(training_inputs, training_outputs, 100)

    print("Ending Weights After Training: ")
    print(neural_network.synaptic_weights)

    inputTestCase = [1, 1]

    print("Considering New Situation: ", inputTestCase[0], inputTestCase[1], end=" ")
    print("New Output data: ", end=" ")
    print(neural_network.think(np.array(inputTestCase)))
    print("Wow, we did it!")


# endregion
if __name__ == '__main__':
    NN_Main()
