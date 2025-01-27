import numpy as np
from os import path, makedirs
from time import time


def makedirs_(dir_path):  # Создает папку только, если она не существует
    if not path.isdir(dir_path):
        makedirs(dir_path)


def ReLU(x):  # Функция активации
    return np.maximum(0, x)


def ReLU_deriv(Z):  # Производная ReLU
    return np.where(Z > 0, 1, 0)


def SoftMax(vect):
    exp_vect = np.exp(vect - np.max(vect))
    return exp_vect / np.sum(exp_vect)

def SoftMax_batch(vect):  # Преобразование батча к вероятностям
    exp_vect = np.exp(vect - np.max(vect, axis=1, keepdims=True))
    return exp_vect / np.sum(exp_vect, axis=1, keepdims=True)


def cross_entropy_batch(p, y, epsilon=1e-12):  # Вычисление расстояния между вероятностями
    p = np.clip(p, epsilon, 1.0)  # Защита от деления на 0
    return -np.sum(np.log(p[np.arange(len(y)), y])) / len(y)


# Преобразование вектора правильных ответов к матрице истинных вероятностей
def y_vector_batch(y, num_classes):
    return np.eye(num_classes)[y]


def num_correct(p, y): # Подсчет верных ответов
    return np.sum(np.argmax(p, axis=1) == y)


class Network:  # Инициализация нейросети
    def __init__(self, network_path, number_neurons=None, train_data_path=None, test_data_path=None,
                 epochs=None, batch_size=None, train_rate=None):  # Все нужные переменные
        self.network_path = network_path + '\\'
        self.number_neurons = number_neurons
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_rate = train_rate

    def weights_bias_init(self):
        self.layers_weights = []  # Инициализация пустых весов
        self.layers_bias = []
        for num_neurons in range(len(self.number_neurons) - 1):  # Присваивание случайных значений для всех слоев
            self.layers_weights.append(np.random.normal(0.0, pow(self.number_neurons[num_neurons], -0.5), (self.number_neurons[num_neurons], self.number_neurons[num_neurons + 1])))
            self.layers_bias.append(np.random.normal(0.0, pow(self.number_neurons[num_neurons + 1], -0.5), (self.number_neurons[num_neurons + 1])))

    def save(self, learn_time=0.0):
        makedirs_(f'{self.network_path}weights')  # Создаем пустые папки весов и смещений
        makedirs_(f'{self.network_path}bias')
        for i in range(len(self.layers_weights)):  # Сохраняем весы
            np.savetxt(f'{self.network_path}weights\\weight{i}.csv', self.layers_weights[i], delimiter=',')
            np.savetxt(f'{self.network_path}bias\\bias{i}.csv', self.layers_bias[i])
        with open(f'{self.network_path}info.txt', 'w', encoding="UTF-8") as inf:
            info = f"path - '{self.network_path}'\nlayers - {self.number_neurons}\ntrain_data_path - '{self.train_data_path}'\n"
            info += f"test_data_path - '{self.test_data_path}'\nepochs - '{self.epochs}'\nbatch - '{self.batch_size}'"
            info += f"\ntrain_rate - '{self.train_rate}'\nlearn time - '{learn_time}'"
            inf.write(info)

    def load(self):
        self.layers_weights = []  # Инициализация пустых весов
        self.layers_bias = []
        i = 0  # Проход по файлам, пока они не закончатся
        while path.isfile(f'{self.network_path}weights\\weight{i}.csv'):
            self.layers_weights.append(np.loadtxt(f'{self.network_path}weights\\weight{i}.csv', dtype='float64', delimiter=','))
            self.layers_bias.append(np.loadtxt(f'{self.network_path}bias\\bias{i}.csv', dtype='float64'))
            i += 1

    def predict(self, image):  # Предсказание нейросети
        for i in range(len(self.layers_bias) - 1):
            image = ReLU(image @ self.layers_weights[i] + self.layers_bias[i])
        image = SoftMax(image @ self.layers_weights[-1] + self.layers_bias[-1])
        return image

    def predict_batch(self, image):  # Предсказание нейросети
        for i in range(len(self.layers_bias) - 1):
            image = ReLU(image @ self.layers_weights[i] + self.layers_bias[i])
        image = SoftMax_batch(image @ self.layers_weights[-1] + self.layers_bias[-1])
        return image

    def test(self):
        test_data = np.loadtxt(self.test_data_path, dtype="float64", delimiter=',')
        y = test_data[:, 0]
        p = self.predict_batch(test_data[:, 1:] / 255)
        return round(num_correct(p, y) / test_data.shape[0] * 100, 5)

    def learn(self):

        # Загрузка базы тренировочных данных
        train_data = np.loadtxt(self.train_data_path, dtype="float64", delimiter=',')
        train_data[:, 1:] /= 255
        print('Начало обучения')
        t = time()  # Начало отчета времени

        for ep in range(self.epochs):
            epoch_loss = 0
            correct_count = 0
            np.random.shuffle(train_data)  # Перемешивание данных

            for i in range(train_data.shape[0] // self.batch_size):

                # Прямое распространение нейросети
                layer = [train_data[i * self.batch_size: (i + 1) * self.batch_size, 1:]]
                y = train_data[i * self.batch_size: (i + 1) * self.batch_size, 0].astype(np.uint8)
                Z = []

                for L in range(len(self.number_neurons) - 2):
                    Z.append(layer[L] @ self.layers_weights[L] + self.layers_bias[L])
                    layer.append(ReLU(Z[L]))
                layer.append(SoftMax_batch(layer[-1] @ self.layers_weights[-1] + self.layers_bias[-1]))


                # Обратное распространение
                dE_dZ = layer[-1] - y_vector_batch(y, self.number_neurons[-1])
                for L in range(len(self.number_neurons) - 2, -1, -1):
                    dE_dW = np.transpose(layer[L]) @ dE_dZ
                    dE_dB = np.sum(dE_dZ, axis=0, keepdims=True)
                    dE_dB.resize((self.number_neurons[L + 1], ))
                    if L > 0:
                        dE_dA = dE_dZ @ np.transpose(self.layers_weights[L])
                        dE_dZ = dE_dA * ReLU_deriv(Z[L - 1])
                    
                    # Корректировка весов
                    self.layers_weights[L] -= self.train_rate * dE_dW
                    self.layers_bias[L] -= self.train_rate * dE_dB


                    epoch_loss += np.sum(cross_entropy_batch(layer[-1], y))
                    correct_count += num_correct(layer[-1], y)

            # Вывод номера эпохи и точности нейросети
            accuracy = correct_count / train_data.shape[0]
            print(f"Epoch {ep+1}/{self.epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.4f}, Test = {self.test()}")

        self.save(time() - t)
        print(f"Нейросеть обучена")


def main():
    digit = Network(r'network\neural_networks\digit_project', [784, 256, 10],
                    r'network\mnist\mnist_train.csv', r'network\mnist\mnist_test.csv',
                    5, 5000, 0.00001)
    #digit.weights_bias_init()
    digit.load()
    digit.learn()
    print(digit.test())


if __name__ == "__main__":
    main()
