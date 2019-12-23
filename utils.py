# @title ###MUST BE RUN AT START {display-mode: "form"}



import numpy as np
import time
from sklearn.utils import shuffle
from scipy.stats import truncnorm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pycm


###############################################################################

def truncatedNormal(mean=0, sd=1, low=0, upp=1):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#%%
def oneHotEncoding(y):
    encoded = to_categorical(y)
    return encoded

s = oneHotEncoding(7)

#%%
def oneHotEncoding2(y):
    zeros = [0,0,0,0,0,0,0,0,0,0]
    zeros[y-1] = 1
    return zeros



#%%


###############################################################################
############################ ACTIVATION FUNCTIONS #############################
###############################################################################

def softmax(x):
    """
    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def relu(x):
    return np.maximum(0.0, x)


def derivativeRelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def derivativeSigmoid(x):
    return sigmoid(x) * (1. - sigmoid(x))


###############################################################################
############################### ERROR FUNCTIONS ###############################
###############################################################################

def calculateCrossEntropyLoss(prob_predictions, targets):
    ce = - targets * np.log(prob_predictions)
    return np.sum(ce, axis=1)


###############################################################################
################################## Analytics ##################################
###############################################################################

def createConfusionMatrix(predictions, targets):
    """
    https://www.pycm.ir/doc/index.html
    """
    return pycm.ConfusionMatrix(actual_vector=targets, predict_vector=predictions)

#%%
tr = oneHotEncoding(7)
print(len(tr))