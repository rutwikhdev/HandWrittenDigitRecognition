import numpy as np
import matplotlib.pyplot as plt
from helper_utils import *


def initi_params(layer_dims):
    np.random.seed(3)
    params = {}
    L = len(layer_dims)           

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(params['b' + str(l)].shape == (layer_dims[l], 1))
  
    return params

"""**********FORWARD PROPAGATION**********"""

def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
  
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "softmax":
    
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = stable_softmax(Z)
    
    elif activation == "relu":
    
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, params):
    caches = []
    A = X
    L = len(params) // 2                 
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, params['W'+str(l)], params["b"+str(l)], 'relu')
        caches.append(cache)

    AL, cache = linear_activation_forward(A, params["W"+str(L)], params["b"+str(L)], 'softmax')
    caches.append(cache)
    assert(AL.shape == (10,X.shape[1]))
    return AL, caches

"""**********BACKPROPAGATION**********"""

def compute_cost(AL, Y):
    #cost = -1 * np.mean(Y*(AL + (-AL.max() - np.log(np.sum(np.exp(AL - AL.max()))))))
    cost = -np.mean(Y.T * np.log(AL.T + 1e-8))
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = (np.sum(dZ, axis=1)).reshape(b.shape[0],b.shape[1]) / m    
    dA_prev = np.dot(W.T, dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dZ = AL - Y                 # derivative of cross entropy wrt activation func(final layer)
    
    current_cache = caches[L-1]
    linear_cache, _ = current_cache
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dZ, linear_cache)
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_params(params, grads, learning_rate=0.05):
    L = len(params) // 2
    for l in range(L):
        params["W" + str(l+1)] = params["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        params["b" + str(l+1)] = params["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return params

"""MODEL"""

def L_layer_model(X, Y, layers_dims, learning_rate = 0.5, num_iterations = 3000, print_cost=True):#lr was 0.009
    costs = []                         # keep track of cost
    
    params = initi_params(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, params)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        params = update_params(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return params

def predict(actual, predictions, total_examples):
    correct = 0
    incorrect = 0

    for i in range(total_examples): # 60000
        for j in range(10): # 10
            if predictions[i][j] == max(predictions[i]):
                predictions[i][j] = 1
            else:
                predictions[i][j] = 0

    for k in range(total_examples):
        for l in range(10):
            if predictions[k][l] == 1:
                if predictions[k][l] == actual[k][l]:
                    correct += 1
                else:
                    incorrect += 1
    print("Total correct predictions: ",correct)
    print("Total incorrect predictions: ", incorrect)

    return correct, incorrect