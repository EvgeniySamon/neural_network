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
    def __init__(self):
        pass


    def init_weights(self, layers):
        self.weights = []  # Инициализация пустых весов
        self.bias = []

        for num in range(len(layers) - 1):  # Присваивание случайных значений для всех слоев
            self.weights.append(
                np.random.normal(
                    0.0, pow(layers[num], -0.5), (layers[num], layers[num + 1])
                    )
                )
            self.bias.append(
                np.random.normal(
                    0.0, pow(layers[num + 1], -0.5), (layers[num + 1])
                    )
                )


    def save(self, save_path):
        makedirs_(f'{save_path}\\weights')  # Создаем пустые папки весов и смещений
        makedirs_(f'{save_path}\\bias')

        for i in range(len(self.weights)):  # Сохраняем весы
            np.savetxt(f'{save_path}\\weights\\weight{i}.csv', self.weights[i], delimiter=',')
            np.savetxt(f'{save_path}\\bias\\bias{i}.csv', self.bias[i])

        with open(f'{save_path}\\info.txt', 'w', encoding="UTF-8") as inf:
            info = f"layers - {list(i.shape for i in self.weights)}"
            inf.write(info)


    def load(self, load_path):
        self.weights = []  # Инициализация пустых весов
        self.bias = []

        i = 0  # Проход по файлам, пока они не закончатся
        while path.isfile(f'{load_path}\\weights\\weight{i}.csv'):
            self.weights.append(np.loadtxt(f'{load_path}\\weights\\weight{i}.csv', dtype='float64', delimiter=','))
            self.bias.append(np.loadtxt(f'{load_path}\\bias\\bias{i}.csv', dtype='float64'))
            i += 1


    def predict(self, data):  # Предсказание нейросети
        for i in range(len(self.bias) - 1):
            data = ReLU(data @ self.weights[i] + self.bias[i])
        return SoftMax(data @ self.weights[-1] + self.bias[-1])


    def predict_batch(self, data):  # Предсказание нейросети
        for i in range(len(self.bias) - 1):
            data = ReLU(data @ self.weights[i] + self.bias[i])
        return SoftMax_batch(data @ self.weights[-1] + self.bias[-1])


    def test(self, data, labels):
        return num_correct(self.predict_batch(data), labels) / data.shape[0] * 100


    def learn(self, data, labels, epochs, batch_size, train_rate, valid_data=None, valid_labels=None):
        print('Начало обучения')
        for ep in range(epochs):
            epoch_loss = 0
            correct_count = 0

            indices = np.arange(data.shape[0])  # Перемешивание данных
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]

            for i in range(data.shape[0] // batch_size):

                # Прямое распространение нейросети
                layer = [data[i * batch_size : (i + 1) * batch_size]]
                y = labels[i * batch_size : (i + 1) * batch_size]
                Z = []

                for L in range(len(self.weights) - 1):
                    Z.append(layer[L] @ self.weights[L] + self.bias[L])
                    layer.append(ReLU(Z[L]))
                layer.append(SoftMax_batch(layer[-1] @ self.weights[-1] + self.bias[-1]))


                # Обратное распространение
                dE_dZ = layer[-1] - y_vector_batch(y, self.weights[-1].shape[1])
                for L in range(len(self.weights) - 1, -1, -1):

                    dE_dW = np.transpose(layer[L]) @ dE_dZ
                    dE_dB = np.sum(dE_dZ, axis=0, keepdims=True)
                    dE_dB.resize(self.bias[L].shape)
                    if L > 0:
                        dE_dA = dE_dZ @ np.transpose(self.weights[L])
                        dE_dZ = dE_dA * ReLU_deriv(Z[L - 1])
                    
                    # Корректировка весов
                    self.weights[L] -= train_rate * dE_dW
                    self.bias[L] -= train_rate * dE_dB

                epoch_loss += np.sum(cross_entropy_batch(layer[-1], y))
                correct_count += num_correct(layer[-1], y)

            # Вывод номера эпохи и точности нейросети
            accuracy = correct_count / data.shape[0] * 100
            print(f'Epoch {ep+1}/{epochs}: Loss = {epoch_loss:.2f}, Train accuracy = {accuracy:.2f}%, Valid accuracy = {self.test(valid_data, valid_labels):.2f}%')  

        print(f'Нейросеть обучена')


def main():
    train = np.loadtxt(r'mnist\mnist_train.csv', dtype='uint8', delimiter=',')
    train_data = train[:, 1:].astype('float64') / 255
    train_labels = train[:, 0]

    test = np.loadtxt(r'mnist\mnist_test.csv', dtype='uint8', delimiter=',')
    test_data = test[:, 1:].astype('float64') / 255
    test_labels = test[:, 0] 


    digit = Network()
    #digit.load(r'neural_networks\digit_test')
    digit.init_weights([784, 256, 10])
    digit.learn(train_data, train_labels, 50, 5000, 0.0001, test_data, test_labels)
    digit.save(r'neural_networks\digit_test')
    print(digit.test(test_data, test_labels))


if __name__ == "__main__":
    main()
