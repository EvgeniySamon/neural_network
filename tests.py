"""     a = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    b = np.array([0, 1, 1, 0])
    digit = Network()
    #digit.load(r'neural_networks\digit_xor')
    digit.init_weights([2, 128, 2])
    digit.learn(a, b, 5000, 4, 0.0001, a, b)
    digit.save(r'neural_networks\digit_test')
    print(digit.test(a, b))
    print(np.argmax(digit.predict(a), axis=1))
 """


import numpy as np


def SoftMax(vect):
    exp_vect = np.exp(vect - np.max(vect))
    return exp_vect / np.sum(exp_vect)

def SoftMax_batch(vect):  # Преобразование батча к вероятностям
    exp_vect = np.exp(vect - np.max(vect, axis=1, keepdims=True))
    return exp_vect / np.sum(exp_vect, axis=1, keepdims=True)


a = np.array([1, 1, 1, 0, 99, -1, 45])
print(SoftMax(a))
print(SoftMax_batch(a))