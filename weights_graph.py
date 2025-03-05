import numpy as np
import matplotlib.pyplot as plt


while(True):
  i = int(input(f'Визуализацию какого слоя весов вы хотите увидеть: '))
  weights_graph = np.loadtxt(f'MNIST_model\\weights\\weight{i - 1}.csv', dtype='float64', delimiter=',')

  #weights1_graph = min_max_scaling(weights1_graph)
  j = int(input(f'Визуализацию с каким по счету нейроном следующего слоя вы хотите увидеть (из {weights_graph.shape[1]}): '))
  plt.imshow(weights_graph[:, j - 1].reshape((int(weights_graph.shape[0] ** (1/2)), int(weights_graph.shape[0] ** (1/2)))), cmap='gray')
  plt.show()
