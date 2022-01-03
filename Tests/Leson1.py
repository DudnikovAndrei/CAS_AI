testsize = 100
import random
def y(x, m, b):
  return x * m + b

import numpy as np
X = np.linspace(0, 10, testsize)
y_above = [y(x, 10, 5) + abs(random.gauss(20,40)) for x in X]
y_below = [y(x, 10, 5) - abs(random.gauss(20,40)) for x in X]
import matplotlib.pyplot as plt
plt.scatter(X, y_above, c='g')
plt.scatter(X, y_below, c='b')
plt.plot(X, y(X, 10, 5),linestyle='solid', c='r', linewidth=3, label='decision boundary')
plt.legend()
plt.show()




import numpy as np
from sklearn.metrics import accuracy_score
class Perceptron(object):

    def __init__(self, dim_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(dim_inputs + 1)  # plus 1 for bias

    def predict_batch(self, inputs):
        res_vector = np.dot(inputs, self.weights[1:]) + self.weights[0]
        activations = [1 if elem > 0 else 0 for elem in res_vector]
        return np.array(activations)

    def predict(self, inputs):
        res = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # self.weights[0] is the bias
        if res > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs  # update weights
                self.weights[0] += self.learning_rate * (label - prediction)  # update bias





x_1 = np.linspace(0, 10, testsize)

x_2 = np.array([y(elem, 10, 5) + abs(random.gauss(20,40)) for elem in x_1])
class_ones = np.column_stack((x_1, x_2))

x_2 = np.array([y(elem, 10, 5) - abs(random.gauss(20,40)) for elem in x_1])
class_zeros = np.column_stack((x_1, x_2))

training_inputs = np.vstack((class_ones, class_zeros))

print(training_inputs.shape)

labels = np.hstack((np.ones(testsize), np.zeros(testsize))).T
labels.shape



perceptron = Perceptron(2, epochs=100, learning_rate=1)
perceptron.train(training_inputs, labels)



# Inference
accur_arr = perceptron.predict_batch(class_ones)
print(accur_arr)



accur_arr2 = perceptron.predict_batch(class_zeros)
from sklearn.metrics import accuracy_score
print(accur_arr2)
print((accuracy_score(np.ones(testsize), accur_arr)+accuracy_score(np.zeros(testsize), accur_arr2))/2)


# weights
perceptron.weights[1:]


# bias
perceptron.weights[0]