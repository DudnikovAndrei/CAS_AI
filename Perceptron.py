import numpy as np
import random
import matplotlib.pyplot as plt
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
        return 1 if res > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs  # update weights
                self.weights[0] += self.learning_rate * (label - prediction)  # update bias

class Samples:
    def __init__(self, nb_samples, fy, gen_mode="linespace", random_mode="gauss"):
        self.nb_samples = nb_samples
        self.fy = fy

        if gen_mode == "linespace":
            self.x1 = np.linspace(0, 10, nb_samples)
        elif gen_mode == "random_simple":
            self.x1 = np.random.random_sample((50)) * 10
        else:
            raise Exception("Unknown gen_mode")

        if random_mode == "gauss":
            self.x2_ones  = np.array([self.fy(x) + abs(random.gauss(20,40)) for x in self.x1])
            self.x2_zeros = np.array([self.fy(x) - abs(random.gauss(20,40)) for x in self.x1])
        elif random_mode == "randint":
            self.x2_ones  = np.array([self.fy(x) + abs(random.randint(1,5)) for x in self.x1])
            self.x2_zeros = np.array([self.fy(x) - abs(random.randint(1,5)) for x in self.x1])
        elif random_mode == "normal":
            self.x2_ones  = np.array([self.fy(x) + abs(np.random.standard_normal(1)) for x in self.x1])
            self.x2_zeros = np.array([self.fy(x) - abs(np.random.standard_normal(1)) for x in self.x1])
        else:
            raise Exception("Unknown random_mode")

        self.class_ones = np.column_stack((self.x1, self.x2_ones))
        self.class_zeros = np.column_stack((self.x1, self.x2_zeros))

        self.training_inputs = np.vstack((self.class_ones, self.class_zeros))

        self.labels = np.hstack((np.ones(nb_samples), np.zeros(nb_samples))).T
        self.labels.shape

        print(self.training_inputs.shape)

        plt.scatter(self.x1, self.x2_ones, c='g')
        plt.scatter(self.x1, self.x2_zeros, c='b')
        plt.plot(self.x1, self.fy(self.x1), linestyle='solid', c='r', linewidth=3, label='decision boundary')
        plt.legend()
        plt.show()

class Expirience:
    def __init__(self, dim_inputs, epochs, nb_samples, lr, fy, gen_mode, random_mode):
        samples = Samples(nb_samples, fy, gen_mode, random_mode)
        perceptron = Perceptron(dim_inputs, epochs, lr)
        perceptron.train(samples.training_inputs, samples.labels)

        # Inference
        accur_arr = perceptron.predict_batch(samples.class_ones)
        print(accur_arr)

        accur_arr2 = perceptron.predict_batch(samples.class_zeros)
        print(accur_arr2)
        #print((accuracy_score(np.ones(nb_samples), accur_arr) + accuracy_score(np.zeros(nb_samples), accur_arr2)) / 2)

        # weights
        print(perceptron.weights[1:])
        # bias
        print(perceptron.weights[0])

def y(x, m, b):
  return x * m + b

def fy_10_5(x):
    return y(x, 10, 5)

def fy_exp(x):
    return y(x*x, 10, 5)

# -------------------
# Exp
# -------------------

#Exp 1
dim_inputs=2
epochs=100
nb_samples=100
lr=0.01
gen_mode="linespace"
random_mode="gauss"
print("------------------------------------------Exp 1--------------------------------------------")
Expirience(dim_inputs, epochs, nb_samples, lr, fy_10_5, gen_mode, random_mode)

#Exp 2
dim_inputs=2
epochs=100
nb_samples=50
lr=0.01
gen_mode="linespace"
random_mode="gauss"
print("------------------------------------------Exp 2--------------------------------------------")
Expirience(dim_inputs, epochs, nb_samples, lr, fy_10_5, gen_mode, random_mode)

#Exp 3
dim_inputs=2
epochs=100
nb_samples=100
lr=0.01
gen_mode="random_simple"
random_mode="gauss"
print("------------------------------------------Exp 3--------------------------------------------")
Expirience(dim_inputs, epochs, nb_samples, lr, fy_10_5, gen_mode, random_mode)

#Exp 4
dim_inputs=2
epochs=100
nb_samples=500
lr=0.01
gen_mode="random_simple"
random_mode="randint"
print("------------------------------------------Exp 4--------------------------------------------")
Expirience(dim_inputs, epochs, nb_samples, lr, fy_10_5, gen_mode, random_mode)

#Exp 5
dim_inputs=2
epochs=100
nb_samples=500
lr=0.5
gen_mode="linespace"
random_mode="gauss"
print("------------------------------------------Exp 5--------------------------------------------")
Expirience(dim_inputs, epochs, nb_samples, lr, fy_exp, gen_mode, random_mode)



