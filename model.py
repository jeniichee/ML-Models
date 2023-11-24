import numpy as np
import matplotlib.pyplot as plt
import math
import util

# delete pls 
from sklearn.linear_model import LinearRegression
import seaborn as sns

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
        "*** YOUR CODE HERE ***"
        self.degree = degree
        self.learning_rate = learning_rate
        self.weights = np.zeros(degree+1)
 
    def get_features(self, x):
        "*** YOUR CODE HERE ***"
        # x^degree for each degree 
        features = [x ** degree for degree in range(self.degree+1)] # +1 dummy value
        return features 

    def get_weights(self):
        "*** YOUR CODE HERE ***"
        return self.weights

    def hypothesis(self, x): # fix 
        "*** YOUR CODE HERE ***"
        features = self.get_features(x) # get list of features 
        weights = self.get_weights() # get list of weights  
        hypothesis = np.sum(features*weights)
        return hypothesis

    def predict(self, x):
        return self.hypothesis(x)

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"
        pred = self.predict(x)
        error = pred - y
        loss = np.sum(error**2)
        return loss

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"
        pred = self.predict(x)
        error = pred - y
        features = self.get_features(x)
        gradient = np.sum(2*(error)*np.array(features)) ##fix 
        return gradient

    def train(self, dataset, evalset = None): # ignore evalset
        "*** YOUR CODE HERE ***"
        np.random.seed(2) 
        self.weights = np.random.randn(self.degree+1)
        xs, ys = dataset.get_all_samples()
               
        for iteration in range(1000):
            for x, y in zip(xs, ys): 
                grad = self.gradient(x,y)
                self.weights -= self.learning_rate*grad
    
# PA4 Q2
def linear_regression():
    "*** YOUR CODE HERE ***"
    # Examples
    # sine_val = util.get_dataset("sine_val")
     
    sine_train = util.get_dataset("sine_train") # load data
    sine_model = PolynomialRegressionModel(degree=1, learning_rate=1e-4) # model
    sine_model.train(sine_train) # train 
    
    # final hypothesis
    hypothesis = sine_model.get_weights()
    print("final hypothesis:" + str(hypothesis))
    
    # average loss 
    avg_loss = sine_train.compute_average_loss(sine_model)
    print("average loss:" + str(avg_loss))
    
    # plot 
    sine_train.plot_data(sine_model)


# PA4 Q3
class BinaryLogisticRegressionModel(Model):
    """
    Binary logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is either 0 or 1.
    The goal is to fit P(y = 1 | x) = hypothesis(x), and to make a 0/1 prediction using the hypothesis.
    """

    def __init__(self, num_features, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"

    def get_features(self, x):
        "*** YOUR CODE HERE ***"

    def get_weights(self):
        "*** YOUR CODE HERE ***"

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"


# PA4 Q4
def binary_classification():
    "*** YOUR CODE HERE ***"


# PA4 Q5
class MultiLogisticRegressionModel(Model):
    """
    Multinomial logistic regression model with image-pixel features (num_features = image size, e.g., 28x28 = 784 for MNIST).
    x is a 2-D image, represented as a list of lists (28x28 for MNIST). y is an integer between 1 and num_classes.
    The goal is to fit P(y = k | x) = hypothesis(x)[k], where hypothesis is a discrete distribution (list of probabilities)
    over the K classes, and to make a class prediction using the hypothesis.
    """

    def __init__(self, num_features, num_classes, learning_rate = 1e-2):
        "*** YOUR CODE HERE ***"

    def get_features(self, x):
        "*** YOUR CODE HERE ***"

    def get_weights(self):
        "*** YOUR CODE HERE ***"

    def hypothesis(self, x):
        "*** YOUR CODE HERE ***"

    def predict(self, x):
        "*** YOUR CODE HERE ***"

    def loss(self, x, y):
        "*** YOUR CODE HERE ***"

    def gradient(self, x, y):
        "*** YOUR CODE HERE ***"

    def train(self, dataset, evalset = None):
        "*** YOUR CODE HERE ***"


# PA4 Q6
def multi_classification():
    "*** YOUR CODE HERE ***"


def main():
    linear_regression()
    binary_classification()
    multi_classification()

if __name__ == "__main__":
    main()
