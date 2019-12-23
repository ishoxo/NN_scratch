
# Utilities

import utils
import numpy as np
from utils import derivativeRelu
from utils import derivativeSigmoid
from utils import truncatedNormal
from utils import oneHotEncoding
from utils import softmax
from utils import relu
from utils import sigmoid
from utils import calculateCrossEntropyLoss
from utils import createConfusionMatrix
from sklearn.model_selection import train_test_split
import time
from sklearn.utils import shuffle
from digit_structure import images
from digit_structure import labels





# %%

class NeuralNetwork:

    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.structure = []
        self.predict_cm = None

        self.train_analytics = {'acc_val': [],
                                'loss_val': [],
                                'acc_train': [],
                                'loss_train': []}

    ################################################################################

    def addInputLayer(self, num_nodes, dropout_prob=0, bias_node=None):
        layer_params = {
            'layer_type': 'input',
            'num_nodes': num_nodes,
            'bias_node': bias_node,
            'dropout_prob': dropout_prob,
            'dropout_vector': None,
            'output_vectors': None
        }
        self.structure.append(layer_params)

    ################################################################################

    def addHiddenLayer(self, num_nodes, activation='relu', dropout_prob=0, bias_node=None):
        layer_params = {
            'layer_type': 'hidden',
            'num_nodes': num_nodes,
            'activation': activation,
            'dropout_prob': dropout_prob,
            'bias_node': bias_node,
            'weight_matrix': self.createWeightMatrices(num_nodes),
            'dropout_vector': None,
            'weight_update': None,
            'velocity': None,
            'r': None,
            's': None,
            't': 0,
            'output_vectors': None
        }
        self.structure.append(layer_params)

    ################################################################################

    def addOutputLayer(self, num_nodes, activation='softmax'):
        layer_params = {
            'layer_type': 'output',
            'num_nodes': num_nodes,
            'activation': activation,
            'weight_matrix': self.createWeightMatrices(num_nodes),
            'dropout_prob': 0,
            'dropout_vector': None,
            'weight_update': None,
            'velocity': None,
            'r': None,
            's': None,
            't': 0,
            'output_vectors': None
        }
        self.structure.append(layer_params)

    ################################################################################

    def createWeightMatrices(self, num_nodes):
        """
        Sources:
        https://keras.io/initializers/
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
        """
        bias = 1 if self.structure[-1]['bias_node'] else 0
        glorot_sd = np.sqrt(2. / (self.structure[-1]['num_nodes'] + num_nodes))
        z = truncatedNormal(mean=0., sd=glorot_sd, low=-2. * glorot_sd, upp=2. * glorot_sd)
        weights = z.rvs((num_nodes, self.structure[-1]['num_nodes'] + bias), random_state=42)
        if bias == 1:
            # initialise biases to 0.0001
            weights[:, -1] = 0.0001
        return weights

    ################################################################################

    def train(self, X_train, y_train, X_val, y_val, epochs, minibatch_size, L2_lambda, alpha, optimiser='sgd_momentum',
              stopping=None):
        """
        """
        # for each epoch
        min_loss_count = 0
        for _ in np.arange(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for layer in self.structure:
                if layer['layer_type'] != 'input':
                    layer['velocity'] = np.zeros((layer['weight_matrix'].shape))
                    layer['s'] = np.zeros((layer['weight_matrix'].shape))
                    layer['r'] = np.zeros((layer['weight_matrix'].shape))
                if layer['dropout_prob'] != 0:
                    self.createDropoutVector(layer)


            # for each minibatch
            for j in np.arange(0, len(X_train), minibatch_size):
                for layer in self.structure[1:]:
                    layer['weight_update'] = np.zeros((layer['weight_matrix'].shape))
                if j + minibatch_size < len(X_train):
                    mini_X = X_train[int(j): int(j + minibatch_size)]
                    mini_y = y_train[int(j): int(j + minibatch_size)]
                else:
                    mini_X = X_train[int(j):]
                    mini_y = y_train[int(j):]

                # for each data point in the minibatch
                self.structure[0]['output_vectors'] = mini_X
                self.forwardProp(mini_X)
                #print(self.structure[-1]['output_vectors'])
                output_errors = self.structure[-1]['output_vectors'] - mini_y
                self.backwardProp(errors=output_errors)

                # update weights using average update from minibatch
                if optimiser == 'sgd_momentum':
                    self.updateWeightsSGD(L2_lambda=L2_lambda,
                                          minibatch_size=len(mini_X),
                                          alpha=alpha)
                elif optimiser == 'adam':
                    self.updateWeightsAdam(p1=0.9,
                                           p2=0.999,
                                           minibatch_size=len(mini_X),
                                           L2_lambda=L2_lambda)

            prob_predictions_train = self.predictDuringTraining(X=X_train,
                                                                y=np.argmax(y_train, axis=1))
            cm_train = createConfusionMatrix(np.argmax(prob_predictions_train, axis=1),
                                             targets=np.argmax(y_train, axis=1))
            acc_train = cm_train.Overall_ACC
            loss_train = np.mean(calculateCrossEntropyLoss(prob_predictions=prob_predictions_train,
                                                           targets=y_train))

            prob_predictions_val = self.predictDuringTraining(X=X_val,
                                                              y=np.argmax(y_val, axis=1))
            cm_val = createConfusionMatrix(np.argmax(prob_predictions_val, axis=1),
                                           targets=np.argmax(y_val, axis=1))
            acc_val = cm_val.Overall_ACC
            loss_val = np.mean(calculateCrossEntropyLoss(prob_predictions=prob_predictions_val,
                                                         targets=y_val))

            self.train_analytics['acc_val'].append(acc_val)
            self.train_analytics['loss_val'].append(loss_val)
            self.train_analytics['acc_train'].append(acc_train)
            self.train_analytics['loss_train'].append(loss_train)

            if np.min(self.train_analytics['loss_val']) == self.train_analytics['loss_val'][-1]:
                min_loss_val = self.train_analytics['loss_val'][-1]
                min_loss_count = 0
                # save model
                pass
            else:
                min_loss_count += 1

            if min_loss_count != 0 and min_loss_count % 10 == 0:
                print('min loss count = %g' % (min_loss_count))

            # stopping after "verbose" epochs with decreasing validation loss.
            # stop if best validation loss not beaten for 100 epochs?
            if stopping:
                verbose, patience, min_delta = stopping
                test_list = self.train_analytics['loss_val'][-verbose:]

                if min_loss_count > 200:
                    print('')
                    print('Early Stopping: no improvement on minimum validation loss')
                    print('')
                    print('epoch = %g/%g    train: (acc, val) = (%g, %g), val: (acc, loss) = (%g, %g)' % (_ + 1,
                                                                                                          epochs,
                                                                                                          acc_train,
                                                                                                          loss_train,
                                                                                                          acc_val,
                                                                                                          loss_val))
                    break
                elif _ > patience and all(
                        test_list[i] - test_list[i + 1] < - min_delta for i in np.arange(len(test_list) - 1)):
                    print('')
                    print('Early Stopping')
                    print('')
                    print('epoch = %g/%g    train: (acc, val) = (%g, %g), val: (acc, loss) = (%g, %g)' % (_ + 1,
                                                                                                          epochs,
                                                                                                          acc_train,
                                                                                                          loss_train,
                                                                                                          acc_val,
                                                                                                          loss_val))
                    break

            if _ % 100 == 99 or _ % 10 == 9 and _ < 100 or _ == 0:
                print('')
                print('epoch: %g/%g    train: (acc, val) = (%g, %g), val: (acc, loss) = (%g, %g)' % (_ + 1,
                                                                                                     epochs,
                                                                                                     acc_train,
                                                                                                     loss_train,
                                                                                                     acc_val,
                                                                                                     loss_val))

    ################################################################################

    def backwardProp(self, errors):
        """
        """
        for i in np.arange(len(self.structure) - 1, 0, -1):
            layer = self.structure[i]
            prev_layer = self.structure[i - 1]

            # derivatives of activation functions
            if layer['activation'] == 'relu':
                derivative_output = derivativeRelu(layer['output_vectors'])
            elif layer['activation'] == 'sigmoid':
                derivative_output = derivativeSigmoid(layer['output_vectors'])
            elif layer['activation'] == 'softmax':
                derivative_output = 1.
            else:
                raise Exception('Activation function not recognised')

            delta = errors * derivative_output

            if prev_layer['bias_node']:
                output_vectors = np.concatenate(
                    (prev_layer['output_vectors'], np.ones((prev_layer['output_vectors'].shape[0], 1))),
                    axis=1)  # change 1. to self.bias?
                errors = np.dot(layer['weight_matrix'].T, delta.T)[:-1, :].T
            else:
                output_vectors = prev_layer['output_vectors']
                errors = np.dot(layer['weight_matrix'].T, delta.T).T

            layer['weight_update'] = np.dot(delta.T, output_vectors)

    ################################################################################

    def updateWeightsSGD(self, L2_lambda, minibatch_size, alpha):
        for i in np.arange(1, len(self.structure)):
            layer = self.structure[i]
            # remove (1. - alpha)?
            velocity = alpha * layer['velocity'] - (1. - alpha) * self.learning_rate * (layer['weight_update']
                                                                                        + L2_lambda * layer[
                                                                                            'weight_matrix']) / minibatch_size
            layer['weight_matrix'] += velocity

            # replace velocity in layer_params dictonary of that layer
            layer['velocity'] = velocity

    ################################################################################

    def updateWeightsAdam(self, p1, p2, minibatch_size, L2_lambda):
        for i in np.arange(1, len(self.structure)):
            layer = self.structure[i]
            g = (layer['weight_update'] + L2_lambda * layer['weight_matrix']) / minibatch_size
            layer['s'] = p1 * layer['s'] + (1 - p1) * g
            layer['r'] = p2 * layer['r'] + ((1 - p2) * g ** 2.)
            layer['t'] += 1.
            s_c = layer['s'] / (1. - p1 ** layer['t'])
            r_c = layer['r'] / (1. - p2 ** layer['t'])
            layer['weight_matrix'] -= self.learning_rate * (s_c / ((r_c) ** 0.5 + np.finfo(np.double).eps))

    ################################################################################

    def createDropoutVector(self, layer):
        dropout_prob = layer['dropout_prob']
        d = np.random.rand(layer['num_nodes'], 1)
        d[d <= dropout_prob] = 0
        d[d > dropout_prob] = 1
        layer['dropout_vector'] = d.T

    ################################################################################

    def forwardProp(self, input_vectors):
        """
        input_vector can be tuple, list or ndarray
        """
        for i in np.arange(1, len(self.structure)):
            layer = self.structure[i]
            prev_layer = self.structure[i - 1]

            if prev_layer['bias_node']:
                input_vectors = np.concatenate((input_vectors, np.ones((input_vectors.shape[0], 1))),
                                               axis=1)  # change 1. to self.bias?

            output_vectors = np.dot(layer['weight_matrix'], input_vectors.T).T

            # activation functions
            if layer['dropout_prob'] != 0:
                if layer['activation'] == 'relu':
                    layer['output_vectors'] = layer['dropout_vector'] * relu(output_vectors)
                elif layer['activation'] == 'softmax':
                    layer['output_vectors'] = layer['dropout_vector'] * softmax(output_vectors.T).T
                elif layer['activation'] == 'sigmoid':
                    layer['output_vectors'] = layer['dropout_vector'] * sigmoid(output_vectors)
                else:
                    raise Exception('Activation function not recognised')
            else:
                if layer['activation'] == 'relu':
                    layer['output_vectors'] = relu(output_vectors)
                elif layer['activation'] == 'softmax':
                    layer['output_vectors'] = softmax(output_vectors.T).T
                elif layer['activation'] == 'sigmoid':
                    layer['output_vectors'] = sigmoid(output_vectors)
                else:
                    raise Exception('Activation function not recognised')

            input_vectors = layer['output_vectors']

    ################################################################################

    def predictDuringTraining(self, X, y):
        """
        """
        # forward prop through layers
        for layer_index in np.arange(1, len(self.structure)):
            layer = self.structure[layer_index]
            prev_layer = self.structure[layer_index - 1]

            if prev_layer['bias_node']:
                X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # change 1. to self.bias?

            X = (np.dot(layer['weight_matrix'], X.T)).T

            # activation functions
            if layer['activation'] == 'relu':
                X = relu(X)
            elif layer['activation'] == 'softmax':
                X = softmax(X.T).T
            elif layer['activation'] == 'sigmoid':
                X = sigmoid(X)
            else:
                raise Exception('Activation function not recognised')
        return X

    def predict(self, X, y, show_probs=True):
        """
        """
        # forward prop through layers
        for layer_index in np.arange(1, len(self.structure)):
            layer = self.structure[layer_index]
            prev_layer = self.structure[layer_index - 1]

            if prev_layer['bias_node']:
                X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)  # change 1. to self.bias?

            X = (np.dot(layer['weight_matrix'], X.T)).T

            # activation functions
            if layer['activation'] == 'relu':
                X = relu(X)
            elif layer['activation'] == 'softmax':
                X = softmax(X.T).T
            elif layer['activation'] == 'sigmoid':
                X = sigmoid(X)
            else:
                raise Exception('Activation function not recognised')

        predictions = np.argmax(prob_predictions, axis=1)
        self.predict_cm = createConfusionMatrix(predictions=predictions,
                                                targets=np.argmax(y_train, axis=1))

        if show_probs == True:
            return X
        else:
            return predictions

    ################################################################################

    def saveModel(self, file_name):
        """
        save self.structure & self.train_analytics
        """
        pass

    def loadModel(self, file_name):
        """
        load self.structure & self.train_analytics into
        """
        pass


################################################################################
################################################################################
################################################################################

if __name__ == "__main__":
    from sklearn import datasets

    #iris = datasets.load_iris()
    #X, y = iris.data, iris.target

    #digits = datasets.load_digits()
    X, y = images, labels
    #print(X.shape, 'xshape here bro')
    #print(y.shape)

    num_classes = len(np.unique(y))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    #print(y_val)
    y_train, y_val = oneHotEncoding(y_train), oneHotEncoding(y_val)

    nn = NeuralNetwork(learning_rate=0.01)
    nn.addInputLayer(len(X[0]), dropout_prob=0, bias_node=True)
    nn.addHiddenLayer(1000, activation='relu', dropout_prob=0, bias_node=True)
    nn.addHiddenLayer(500, activation='relu', dropout_prob=0, bias_node=True)
    nn.addHiddenLayer(80, activation='relu', dropout_prob=0, bias_node=True)
    nn.addOutputLayer(num_classes)

    epochs = 500
    minibatch_size = 128
    L2_lambda = 0.1
    alpha = 0.5
    optimiser = 'adam'
    verbose = 3
    patience = 500
    min_delta = 0.

    start = time.time()
    nn.train(X_train=X_train,
             y_train=y_train,
             X_val=X_val,
             y_val=y_val,
             epochs=epochs,
             L2_lambda=L2_lambda,
             minibatch_size=minibatch_size,
             alpha=alpha,
             optimiser=optimiser,
             stopping=(verbose, patience, min_delta))
    train_time = time.time() - start

    #    start = time.time()
    #    nn.predict(X, y, show_probs=True)
    #    predict_time = time.time() - start

    print('')
    print('train time taken = %g' % (train_time))

    print('')
    print(nn.train_analytics['loss_val'])
#    print('')
#    print('predict time taken = %g' % (predict_time))

