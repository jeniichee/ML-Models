import numpy as np
import matplotlib.pyplot as plt
import math
import util

class Model:
    """
    Abstract class for a machine learning model.
    """
    
    def get_features(self, x_input):
        pass

    def get_weights(self):
        pass

    def hypothesis(self, x):
        pass

    def predict(self, x):
        pass

    def loss(self, x, y):
        pass

    def gradient(self, x, y):
        pass

    def train(self, dataset):
        pass


# PA4 Q1
class PolynomialRegressionModel(Model):
    """
    Linear regression model with polynomial features (powers of x up to specified degree).
    x and y are real numbers. The goal is to fit y = hypothesis(x).
    """

    def __init__(self, degree = 1, learning_rate = 1e-3):
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = np.zeros(degree+1)
 
    def get_features(self, x):
        # x^degree for each degree 
        features = [x ** degree for degree in range(self.degree+1)] # +1 dummy value
        return features 

    def get_weights(self):
        return self.weights

    def hypothesis(self, x): # fix 
        return np.dot(self.get_features(x), self.weights)

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        pred = self.predict(x)
        error = pred - y
        loss = np.sum(error**2)
        return loss

    def gradient(self, x, y):
        pred = self.predict(x)
        error = pred - y
        features = self.get_features(x)
        gradient = np.sum(2*(error)*np.array(features)) # doubt
        return gradient

    def train(self, dataset, evalset = None):
        np.random.seed(2) 
        self.weights = np.random.randn(self.degree+1)
        xs, ys = dataset.get_all_samples()
        losses = []
               
        for iteration in range(1000):
            for x, y in zip(xs, ys): 
                grad = self.gradient(x,y)
                self.weights -= self.learning_rate*grad
                
                if evalset is not None and iteration % evalset == 0: 
                    losses.append(dataset.compute_average_loss(self))   
        
        return losses
            
# PA4 Q2
def linear_regression():
    sine_train = util.get_dataset("sine_train") # load training data
    sine_model = PolynomialRegressionModel(degree=1, learning_rate=1e-4) # model
    losses = sine_model.train(sine_train, 100)
    
    # final hypothesis
    hypothesis = sine_model.get_weights()
    print("final hypothesis:" + str(hypothesis))
    
    # average loss 
    avg_loss = sine_train.compute_average_loss(sine_model)
    print("average loss:" + str(avg_loss))
    
    # plot hypothesis
    sine_train.plot_data(sine_model)
    
    # plot loss
    title = "Loss Curve"
    sine_train.plot_loss_curve(np.arange(len(losses)), losses, title)
    
    # hyperparameter search
    sine_val = util.get_dataset("sine_val") # load validation data
    degrees = [1, 2, 3, 4, 5]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    
    best_combo = None
    least_avg_val_loss = float('inf')
    
    for degree in degrees: 
        for lr in lrs: 
            # liner regression on training dataset
            sine_model1 = PolynomialRegressionModel(degree=degree, learning_rate=lr) 
            # losses = sine_model.train(sine_train, 100)
            avg_loss_train = sine_train.compute_average_loss(sine_model1) # training set loss
            avg_val_loss = sine_val.compute_average_loss(sine_model1) # validation set loss

            print(f"degree: {degree}, rate: {lr}")
            print(f"average training set loss: {avg_loss_train}")
            print(f"average validation set loss: {avg_val_loss}\n")

            if avg_val_loss < least_avg_val_loss:
                least_avg_val_loss = avg_val_loss
                best_combo = (degree, lr)

    print("best combo:")
    print(f"degree: {best_combo[0]}, learning rate: {best_combo[1]}")
    print(f"lowest average validation loss: {least_avg_val_loss}")

# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate = 1e-2):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.weights = np.zeros(num_features+1) # dummy bias >:o
    
    def get_features(self, x):
        # x is 2D, return 1D 
        features = []
        for row in x: 
            for pixel in row: 
                features.append(pixel)
        
        return features 

    def get_weights(self):
        return self.get_weights()

    def hypothesis(self, x):
        z = (np.dot(self.get_features(x), self.get_weights()))
        return 1/(1+math.e**(-z)) # sigmoid

    def predict(self, x):
        if self.hypothesis(x)>= 0.5:
            return 1 
        else:
            return 0 

    def loss(self, x, y):
        hx = self.hypothesis(x)
        return (-y*np.log(hx)-(1-y)*np.log(1-hx))

    def gradient(self, x, y):
        # list of partial derivatives of the loss function with respect to each weight.
        hx = self.hypothesis(x) 
        features = self.get_features(x)       
        return (hx-y)*np.array(features) 

    def train(self, dataset, evalset = None):
        np.random.seed(2) 
        self.weights = np.random.randn(self.num_features+1)
        avg_loss = None
        
        for iteration in range(100000):
            total = np.zeros(self.num_features + 1)
            for x, y in dataset:
                grad = self.gradient(x, y)
                total += grad

            # update weights
            self.weights -= self.learning_rate * total / len(dataset)

            if evalset is not None and iteration % evalset == 0:
                avg_loss = np.mean([self.loss(x, y) for x, y in evalset])
                print(f"iteration {iteration}, average loss: {avg_loss}")

# PA4 Q4
def binary_classification():
    return

# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        return

    def get_features(self, x):
        return

    def get_weights(self):
        return

    def hypothesis(self, x):
        return

    def predict(self, x):
        return

    def loss(self, x, y):
        return

    def gradient(self, x, y):
        return

    def train(self, dataset, evalset = None):
        return


# PA4 Q6
def multi_classification():
    return


def main():
    linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()
