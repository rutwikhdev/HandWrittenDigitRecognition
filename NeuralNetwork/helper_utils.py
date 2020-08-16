import numpy as np
import matplotlib.pyplot as plt

def print_image(image, value):
    image = image.reshape(28,28)
    print("The handwritten digit is: ", value)
    plt.imshow(image)
    plt.show()

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
    return dZ

def stable_softmax(Z):
	expZ = np.exp(Z - np.max(Z))
	return expZ / expZ.sum(axis=0, keepdims=True) , Z

def one_hot_matrix(Y):
    Y = Y.astype(np.int).T
    hot = np.zeros((Y.size, int(Y.max()+1)))
    hot[np.arange(Y.size), Y] = 1
    return hot.T

def one_hot_matrix2(Y):
    b = np.zeros((Y.size, Y.max()+1.))
    b[np.arange(Y.size),Y] = 1.
    print(b)
    return b