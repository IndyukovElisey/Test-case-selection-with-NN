from __future__ import print_function
from unicodedata import name

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from random import random, randint
import cProfile
import pstats

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *

import sys
import io
import ast

np.set_printoptions(threshold=sys.maxsize)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("TensorFlow version:", tf.__version__)
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


def clicked():
    print('clicked')


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Тестовый набор')
        self.setGeometry(300,250,350,200)
        self.initUI()
    
    def initUI(self):
        # self.text_edit = QtWidgets.QTextEdit(self)
        # self.setCentralWidget(self.text_edit)
        # self.createMenuBar()
        
        self.label = QtWidgets.QLabel(self)
        self.label.setText('label')
        self.label.move(50,50)
        
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('button')
        self.b1.clicked.connect(self.clicked)
    
    def clicked(self):
        self.label.setText('pressed 1111111111111111111111')
        self.update()
    
    def update(self):
        self.label.adjustSize()    
    
    def createMenuBar(self):
       self.menuBar = QMenuBar(self) 
       self.setMenuBar(self.menuBar)
       
       self.fileMenu = QMenu("&Файл", self)
       self.menuBar.addMenu(self.fileMenu)
 

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        loadUi("NN.ui",self)
        self.btn_createNN.clicked.connect(self.buildCreditNN)
        self.btn_sort.clicked.connect(self.rankAttrCredit)
        self.btn_generate.clicked.connect(self.generateTestCases)
        self.btn_test.clicked.connect(self.testCredit)
        self.btn_page1.clicked.connect(lambda: self.verticalStackedWidget.setCurrentWidget(self.page_1))
        self.btn_page2.clicked.connect(lambda: self.verticalStackedWidget.setCurrentWidget(self.page_2))
        self.btn_page3.clicked.connect(lambda: self.verticalStackedWidget.setCurrentWidget(self.page_3))
        self.btn_page4.clicked.connect(lambda: self.verticalStackedWidget.setCurrentWidget(self.page_4))
        self.action1.triggered.connect(self.aboutDialog)
        self.show()
 
    def buildCreditNN(self):
        train_input = []
        train_output = []
        test_input = []
        test_output = []
        
        epochs_number = int(self.epochs.text())
        train_number = int(self.train.text()) * 2
        test_number = int(self.train.text())
        output_class_number = int(self.range.text())
        out = io.StringIO()

        sys.stdout = out
        
        for i in range(train_number + test_number):
            citizenship = randint(0, 1)
            state = randint(0, 1)
            age = randint(1, 100)
            sex = randint(0, 1)
            region = randint(0, 6)
            income_class = randint(0, 199)
            dependents_number = randint(0, 4)
            marital_status = randint(0, 1)

            credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                        dependents_number, marital_status)
            age = normalize(age, 1, 100)
            region = normalize(region, 0, 6)
            income_class = normalize(income_class, 0, 199)
            dependents_number = normalize(dependents_number, 0, 4)
            credit_limit = normalize(credit_limit, 0, 20000)
            if credit_limit != 0:
                credit_limit = int(credit_limit // 0.1) + 1

            if i < train_number:
                train_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
                train_output.append(credit_limit)
            else:
                test_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
                test_output.append(credit_limit)

        # print(attributes)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, input_dim = 8, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(output_class_number+1)
        ])

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        model.fit(train_input, train_output, epochs=epochs_number)
        model.save('saved_model/credit_model.h5')
        model.summary()
        
        self.result_createNN.setText(out.getvalue())

        test_loss, test_acc = model.evaluate(test_input, test_output, verbose=2)
        self.result_accurasy.setText(f'Точность нейронной сети на тестовом наборе: {test_acc}')
        
        sys.stdout = sys.__stdout__
 
    def rankAttrCredit(self):
        # rank NN input attributes based on weight values
        new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
        # new_model.summary()
        
        # get weights
        weights0 = new_model.layers[0].get_weights()[0]
        # weights1 = new_model.layers[1].get_weights()[0]
        # weights2 = new_model.layers[2].get_weights()[0]

        weight_rank = []

        while weights0.size > len(weight_rank):
            min = 999
            xmin = -1
            ymin = -1
            for ix, iy in np.ndindex(weights0.shape):
                    if abs(weights0[ix][iy]) < abs(min) and (weights0[ix][iy] != weight_rank).all():
                        min = weights0[ix][iy]
                        xmin = ix
                        ymin = iy
            weight_rank.append([xmin, ymin, min])
        
        input_rank = []
        while len(weight_rank) > 0:
            if [i[0] for i in weight_rank].count(weight_rank[0][0]) == 1:
                input_rank.append(weight_rank[0][0])
            weight_rank.pop(0)
        attributes = ['citizenship', 'state', 'age', 'sex', 'region', 'income_class', 'dependents_number', 'marital_status']
        for i in range(len(input_rank)):
            # print(attributes[i])
            input_rank[i]=attributes[input_rank[i]]
        
        self.result_sort.setText(f'{input_rank}')
 
    def generateTestCases(self):
        # generate test cases based on ranked list of input attributes
        
        rank_list = ast.literal_eval(self.result_sort.toPlainText())
        test_cases = []
        # [name, number of possible values, lower limit, upper limit]
        attribute_limits = np.array([['citizenship', 2, 0, 1],
                            ['state', 2, 0, 1],
                            ['age', 3, 1, 100],
                            ['sex', 2, 0, 1],
                            ['region', 4, 0, 6],
                            ['income_class', 3, 0, 199],
                            ['dependents_number', 5, 0, 4],
                            ['marital_status', 2, 0, 1]])
        test_number = 1
        
        # delete 3 least important attributes
        for i in range(len(rank_list)):
            if i < 3:
                x, = np.argwhere(attribute_limits == rank_list[i])
                attribute_limits = np.delete(attribute_limits,x[0],0)

        # attribute names in first row of suite
        test_cases.append([attribute_limits[0][0], attribute_limits[1][0],
                        attribute_limits[2][0],attribute_limits[3][0],
                        attribute_limits[4][0]])
        
        # count number of test cases based on attributes' number of possible values
        for i in range(len(attribute_limits)):
            test_number *= int(attribute_limits[i][1])
        
        # generate test cases
        
        for i in range(test_number):
            test_case = []
            for j in attribute_limits:
                attr = randint(int(j[2]), int(j[3]))
                test_case.append(attr)
            test_cases.append(test_case)  
            
        # test_number = 165 
        # for i in range(test_number):
        #     test_case = []
        #     for j in range(len(attribute_limits)):
        #         repeat = 1
        #         rotate = 1
        #         for k in range(j):
        #             repeat *= int(attribute_limits[k][1])
        #         for k in range(j+1):
        #             rotate *= int(attribute_limits[k][1])
        #         attr = round(((int(attribute_limits[j][3]) - int(attribute_limits[j][2])) / int(attribute_limits[j][1])) * (((i % rotate))//repeat) + 0.0000005) + int(attribute_limits[j][2])
        #         if i == (144 / 2) + 1:
        #             pass
        #         test_case.append(attr)
        #     test_cases.append(test_case)  
        
        attribute_limits = np.array([['citizenship', 2, 0, 1],
                            ['state', 2, 0, 1],
                            ['age', 3, 1, 100],
                            ['sex', 2, 0, 1],
                            ['region', 4, 0, 6],
                            ['income_class', 3, 0, 199],
                            ['dependents_number', 5, 0, 4],
                            ['marital_status', 2, 0, 1]])
        
        # insert back deleted attributes with value '1'        
        z = np.zeros((len(test_cases), 3), dtype=int)
        z = z.astype(np.dtype('U20'))
        expanded_test_cases = np.append(test_cases, z, axis=1)

        for i in range(5):
            x, = np.argwhere(attribute_limits == test_cases[0][i])
            attribute_limits = np.delete(attribute_limits,x[0],0)

        expanded_test_cases[0][5] = str(attribute_limits[0][0])
        expanded_test_cases[0][6] = str(attribute_limits[1][0])
        expanded_test_cases[0][7] = str(attribute_limits[2][0])
        
        np.savetxt('test suite.txt', expanded_test_cases, fmt='%s')
        self.result_generate.setText(str(expanded_test_cases))
 
    def testCredit(self):
        # compares mutated and original version with given test cases
        test_cases = np.loadtxt('test suite.txt', dtype=np.dtype('U20'))
        
        error_found = 0
        test_number = 0
        cases_found_error = ''
        
        attribute_limits = np.array([['citizenship', 2, 0, 1],
                            ['state', 2, 0, 1],
                            ['age', 3, 1, 100],
                            ['sex', 2, 0, 1],
                            ['region', 7, 0, 6],
                            ['income_class', 4, 0, 199],
                            ['dependents_number', 5, 0, 4],
                            ['marital_status', 2, 0, 1]])
        
        for i in range(1, len(test_cases)):
            attributes = {}
            test_number += 1
            
            for j in range(8):
                attributes[test_cases[0][j]] = int(test_cases[i][j])
                
            original_result1, original_result2 = creditApproval(**attributes)
            if self.error.currentIndex() == 0:
                mutated_result1, mutated_result2 = mutatedCreditApproval1(**attributes)
            elif self.error.currentIndex() == 1:
                mutated_result1, mutated_result2 = mutatedCreditApproval2(**attributes)
            elif self.error.currentIndex() == 2:
                mutated_result1, mutated_result2 = mutatedCreditApproval3(**attributes)
            elif self.error.currentIndex() == 3:
                mutated_result1, mutated_result2 = mutatedCreditApproval4(**attributes)
            elif self.error.currentIndex() == 4:
                mutated_result1, mutated_result2 = mutatedCreditApproval5(**attributes)
            elif self.error.currentIndex() == 5:
                mutated_result1, mutated_result2 = mutatedCreditApproval6(**attributes)
            elif self.error.currentIndex() == 6:
                mutated_result1, mutated_result2 = mutatedCreditApproval7(**attributes)
            elif self.error.currentIndex() == 7:
                mutated_result1, mutated_result2 = mutatedCreditApproval8(**attributes)
            elif self.error.currentIndex() == 8:
                mutated_result1, mutated_result2 = mutatedCreditApproval9(**attributes)
            else:
                mutated_result1, mutated_result2 = mutatedCreditApproval10(**attributes)
                
            if original_result2 != mutated_result2:
                cases_found_error += f"Кейс №{i} обнаружил ошибку\n"
                error_found += 1
        # print(f"{test_number} tests, {error_found} errors, error rate: {error_found/test_number}")
        self.result_test.setText(cases_found_error)
        self.test_number.setText(str(test_number))
        self.test_error_found.setText(str(error_found))
        self.test_error_rate.setText(f'{error_found/test_number*100}%')
 
    def aboutDialog(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("О программе")
        dlg.setText("Эта программа предназначена для отбора и генерации тестовых кейсов с применением нейронной сети\nРазработчик: Индюков Е.П.")
        button = dlg.exec()

        if button == QMessageBox.Ok:
            print("OK!")
 
    def update(self):
        self.label.adjustSize()  
        
def application():
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    sys.exit(app.exec_())


def xor():
    startTime = datetime.now()
    # Define our training input and output data with type 16 bit float
    # Each input maps to an output
    startTime = datetime.now()
    X = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=tf.float16)
    Y = tf.constant([[0], [1], [1], [0]], dtype=tf.float16)

    # Create a new Sequential Model
    model = keras.Sequential()

    # Add our layers

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=2,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        10000,  # Amount of Neurons
        input_dim=2000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        10000,  # Amount of Neurons
        input_dim=10000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=10000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        2000,  # Amount of Neurons
        input_dim=2000,  # Define an input dimension because this is the first layer
        activation='relu'  # Use relu activation function because all inputs are positive
    ))

    model.add(layers.Dense(
        1,  # Amount of Neurons. We want one output
        activation='sigmoid'  # Use sigmoid because we want to output a binary classification
    ))

    # Compile our layers into a model

    model.compile(
        loss='mean_squared_error',  # The loss function that is being minimized
        optimizer='adam',  # Our optimization function
        metrics=['binary_accuracy']  # Metrics are different values that you want the model to track while training
    )

    # Our function to take in two numerical inputs and output the relevant boolean
    def cleanPredict(a, b):
        inputTens = tf.constant([[a, b]])
        return round(model.predict(inputTens)[0][0]) == 1  # model.predict(input) yields a 2d tensor

    print(cleanPredict(1, 0))  # Will yield a random value because model isn't yet trained

    model.fit(
        X,  # Input training data
        Y,  # Output training data
        epochs=10,  # Amount of iterations we want to train for
        verbose=1  # Amount of detail you want shown in terminal while training
    )

    print(cleanPredict(1, 0))  # Should Yield True
    print("Time taken:", datetime.now() - startTime)


def digits():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()
    predictions

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test, verbose=2)


class Model(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=units,
                                            activation=tf.nn.relu,
                                            kernel_initializer=tf.random.normal,
                                            bias_initializer=tf.random.normal)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        # For Keras layers/models, implement `call` instead of `__call__`.
        x = x[:, tf.newaxis]
        x = self.dense1(x)
        x = self.dense2(x)
        return tf.squeeze(x, axis=1)


def quadratic():
    matplotlib.rcParams['figure.figsize'] = [9, 6]

    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        y = x ** 2 + 2 * x - 5
        return y

    y = f(x) + tf.random.normal(shape=[201])

    model = Model(64)

    plt.plot(x.numpy(), y.numpy(), '.', label='data')
    plt.plot(x, f(x), label='Ground truth')
    plt.plot(x, model(x), label='Untrained predictions')
    plt.title('Before training')
    plt.legend()
    plt.show()

    variables = model.variables

    optimizer = tf.optimizers.SGD(learning_rate=0.01)

    for step in range(1000):
        with tf.GradientTape() as tape:
            prediction = model(x)
            error = (y - prediction) ** 2
            mean_error = tf.reduce_mean(error)
        gradient = tape.gradient(mean_error, variables)
        optimizer.apply_gradients(zip(gradient, variables))

        if step % 100 == 0:
            print(f'Mean squared error: {mean_error.numpy():0.3f}')

    plt.plot(x.numpy(), y.numpy(), '.', label="data")
    plt.plot(x, f(x), label='Ground truth')
    plt.plot(x, model(x), label='Trained predictions')
    plt.title('After training')
    plt.legend()
    plt.show()

    new_model = Model(64)
    new_model.compile(
        loss=tf.keras.losses.MSE,
        optimizer=tf.optimizers.SGD(learning_rate=0.01))

    history = new_model.fit(x, y,
                            epochs=100,
                            batch_size=32,
                            verbose=0)

    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylim([0, max(plt.ylim())])
    plt.ylabel('Loss [Mean Squared Error]')
    plt.title('Keras training progress')
    plt.show()


def clothesBuild():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5, 5, i + 1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(train_images[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[train_labels[i]])
    # plt.show()

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)
    model.save('saved_model/my_model.h5')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)


def plot_image(i, predictions_array, true_label, img):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10), labels=class_names, rotation=45)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def clothesTest():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    plt.figure(figsize=(10, 10))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    predictions[0]

    np.argmax(predictions[0])
    test_labels[0]

    i = 0
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    i = 12
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()

    # Plot the first X test images, their predicted labels, and the true labels.
    # Color correct predictions in blue and incorrect predictions in red.
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()

    # Grab an image from the test dataset.
    img = test_images[1]

    print(img.shape)

    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))

    print(img.shape)

    predictions_single = probability_model.predict(img)

    print(predictions_single)

    plot_value_array(1, predictions_single[0], test_labels)
    _ = plt.xticks(range(10), class_names, rotation=45)
    plt.show()

    np.argmax(predictions_single[0])


def clothesPredict(i):
    if i < 0 or i > 9999:
        print('out of range')
        return

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    test_images = test_images / 255.0

    new_model = tf.keras.models.load_model('saved_model/my_model.h5')
    new_model.summary()

    loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    print(new_model.predict(test_images).shape)

    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])

    predictions = probability_model.predict(test_images)

    print(predictions[i])

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(1, 2, 2)
    plot_value_array(i, predictions[i], test_labels)
    plt.show()


def creditApproval(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval1(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5: ##################
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval2(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 4 or region == 5: ###############################
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval3(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age > 18: #############################
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval4(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 25: ##############################################
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval5(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 1: ################################
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval6(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 1: ##################################
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval7(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3: ##############################
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval8(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 1 or region == 2: #############################
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit

def mutatedCreditApproval9(citizenship = 0, state = 0, age = 20, sex = 0, region = 3, income_class = 2, dependents_number = 1, marital_status = 0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 5000 ##############################
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 12 * income_class
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit


def mutatedCreditApproval10(citizenship, state, age, sex, region, income_class, dependents_number, marital_status):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 15 * income_class

                if state == 0:
                    if region == 3 or region == 4:
                        credit_limit *= 2
                    else:
                        credit_limit *= 1.5
                else:
                    credit_limit *= 1.1

                if marital_status == 0:
                    if dependents_number > 0:
                        credit_limit += 200 * dependents_number
                    else:
                        credit_limit += 500
                else:
                    credit_limit += 1000

                if sex == 0:
                    credit_limit += 500
                else:
                    credit_limit += 1000
            else:
                credit_limit = 1000 + 2 * income_class #######################
                if marital_status == 0:
                    if dependents_number > 2:
                        credit_limit += 100 * dependents_number
                    else:
                        credit_limit += 100
                else:
                    credit_limit += 300

                if sex == 0:
                    credit_limit += 100
                else:
                    credit_limit += 200
    if credit_limit == 0:
        credit_approved = 1
    else:
        credit_approved = 0

    return credit_approved, credit_limit


def normalize(x, min, max):
    # normalize between 0 and 1
    return (x - min) / (max - min)


def denormalize(x, min, max):
    return round(x*(max-min)+min)


def buildCreditNN(output_class_number):
    train_input = []
    train_output = []
    test_input = []
    test_output = []
    epochs_number = 1000
    train_number = 1000
    test_number = 1000
    
    for i in range(train_number + test_number):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = randint(0, 6)
        income_class = randint(0, 199)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 199)
        dependents_number = normalize(dependents_number, 0, 4)
        credit_limit = normalize(credit_limit, 0, 20000)
        if credit_limit != 0:
            credit_limit = int(credit_limit // 0.1) + 1

        if i < train_number:
            train_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
            train_output.append(credit_limit)
        else:
            test_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
            test_output.append(credit_limit)

    # print(attributes)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim = 8, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(output_class_number+1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_input, train_output, epochs=epochs_number)
    model.save('saved_model/credit_model.h5')

    model.summary()

    test_loss, test_acc = model.evaluate(test_input, test_output, verbose=2)
    print('\nTest accuracy:', test_acc)


def predictCreditNN():
    # randomly generate input values, normalize them and predict output
    test_input = []
    test_output = []

    for i in range(100):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = randint(0, 6)
        income_class = randint(0, 199)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 199)
        dependents_number = normalize(dependents_number, 0, 4)
        credit_limit = normalize(credit_limit, 0, 20000)
        if credit_limit != 0:
            credit_limit = int(credit_limit // 0.1) + 1

        test_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
        test_output.append(credit_limit)

    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    new_model.summary()
    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_input)

    for i in range(100):
        print(np.argmax(predictions[i]), test_output[i], test_input[i], predictions[i])


def inputPredictCreditNN():
    # input attributes, normalize them and predict output with NN
    citizenship = int(input('citizenship'))
    state = int(input('state'))
    age = int(input('age'))
    sex = int(input('sex'))
    region = int(input('region'))
    income_class = int(input('income_class'))
    dependents_number = int(input('dependents_number'))
    marital_status = int(input('marital_status'))

    output = creditApproval(citizenship, state, age, sex, region, income_class,
                            dependents_number, marital_status)[1]
    print('output: ', output)
    output = normalize(output, 0, 18000)
    if output != 0:
        output = int(output // 0.1) + 1
    print('output: ', output)

    age = normalize(age, 1, 100)
    region = normalize(region, 0, 6)
    income_class = normalize(income_class, 0, 199)
    dependents_number = normalize(dependents_number, 0, 4)
    test_input = [citizenship, state, age, sex, region, income_class,
                  dependents_number, marital_status]
    test_input = (np.expand_dims(test_input, 0))

    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    # new_model.summary()
    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_input)
    print('prediction: ', np.argmax(predictions[0]), predictions[0])


def testPredictCredit():
    # predicts program output and compares with actual one
    test_input = []
    test_number = 100

    for i in  range(test_number):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = 2
        income_class = randint(0, 199)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 199)
        dependents_number = normalize(dependents_number, 0, 4)

        test_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])

    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    new_model.summary()
    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_input)

    for i in range(test_number):
        citizenship = test_input[i][0]
        state = test_input[i][1]
        age = denormalize(test_input[i][2],1,100)
        sex = test_input[i][3]
        region = denormalize(test_input[i][4],0,6)
        income_class = denormalize(test_input[i][5],0,199)
        dependents_number = denormalize(test_input[i][6],0,4)
        marital_status = test_input[i][7]
        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        credit_limit = normalize(credit_limit, 0, 20000)
        if credit_limit != 0:
            credit_limit = int(credit_limit // 0.1) + 1

        credit_approved2, credit_limit2 = mutatedCreditApproval1(citizenship, state, age, sex, region, income_class,
                                                                dependents_number, marital_status)
        credit_limit2 = normalize(credit_limit2, 0, 20000)
        if credit_limit2 != 0:
            credit_limit2 = int(credit_limit2 // 0.1) + 1

        if np.argmax(predictions[i]) == credit_limit2:
            if credit_limit2 == credit_limit:
                # print('true positive')
                pass
            else:
                # print('false positive')
                pass
        else:
            if credit_limit2 != credit_limit:
                print('true negative+++++')
            else:
                print('false negative')
                pass


def saveTestCases():
    with open('readme.txt', 'w') as f:
        f.write('Create a new text file!11111111')


def testCredit():
    # compares mutated and original version with given test cases
    test_cases = np.loadtxt('test suite.txt', dtype=np.dtype('U20'))
    
    error_found = 0
    test_number = 0

    attribute_limits = np.array([['citizenship', 2, 0, 1],
                        ['state', 2, 0, 1],
                        ['age', 3, 1, 100],
                        ['sex', 2, 0, 1],
                        ['region', 7, 0, 6],
                        ['income_class', 4, 0, 199],
                        ['dependents_number', 5, 0, 4],
                        ['marital_status', 2, 0, 1]])
    
    for i in range(1, len(test_cases)):
        attributes = {}
        test_number += 1
        
        for j in range(8):
            attributes[test_cases[0][j]] = int(test_cases[i][j])
            
        original_result1, original_result2 = creditApproval(**attributes)
        mutated_result1, mutated_result2 = mutatedCreditApproval1(**attributes)

        if original_result2 != mutated_result2:
            print(f"error found: {i}")
            error_found += 1
    print(f"{test_number} tests, {error_found} errors, error rate: {error_found/test_number}")


def rankAttrCredit():
    # rank NN input attributes based on weight values
    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    new_model.summary()
    # get weights
    weights0 = new_model.layers[0].get_weights()[0]
    weights1 = new_model.layers[1].get_weights()[0]
    # weights2 = new_model.layers[2].get_weights()[0]

    weight_rank = []

    while weights0.size > len(weight_rank):
        min = 999
        xmin = -1
        ymin = -1
        for ix, iy in np.ndindex(weights0.shape):
                if abs(weights0[ix][iy]) < abs(min) and (weights0[ix][iy] != weight_rank).all():
                    min = weights0[ix][iy]
                    xmin = ix
                    ymin = iy
        weight_rank.append([xmin, ymin, min])
    
    input_rank = []
    while len(weight_rank) > 0:
        if [i[0] for i in weight_rank].count(weight_rank[0][0]) == 1:
            input_rank.append(weight_rank[0][0])
        weight_rank.pop(0)
    attributes = ['citizenship', 'state', 'age', 'sex', 'region', 'income_class', 'dependents_number', 'marital_status']
    for i in range(len(input_rank)):
        # print(attributes[i])
        input_rank[i]=attributes[input_rank[i]]
    return input_rank


def generateTestCases(rank_list):
    # generate test cases based on ranked list of input attributes
    
    test_cases = []
    # [name, number of possible values, lower limit, upper limit]
    attribute_limits = np.array([['citizenship', 2, 0, 1],
                        ['state', 2, 0, 1],
                        ['age', 3, 1, 100],
                        ['sex', 2, 0, 1],
                        ['region', 4, 0, 6],
                        ['income_class', 3, 0, 199],
                        ['dependents_number', 5, 0, 4],
                        ['marital_status', 2, 0, 1]])
    test_number = 1
    
    # delete 3 least important attributes
    for i in range(len(rank_list)):
        if i < 3:
            x, = np.argwhere(attribute_limits == rank_list[i])
            attribute_limits = np.delete(attribute_limits,x[0],0)

    # attribute names in first row of suite
    test_cases.append([attribute_limits[0][0], attribute_limits[1][0],
                      attribute_limits[2][0],attribute_limits[3][0],
                      attribute_limits[4][0]])
    
    # count number of test cases based on attributes' number of possible values
    for i in range(len(attribute_limits)):
        test_number *= int(attribute_limits[i][1])
    
    # generate test cases
    
    for i in range(test_number):
        test_case = []
        for j in attribute_limits:
            attr = randint(int(j[2]), int(j[3]))
            test_case.append(attr)
        test_cases.append(test_case)  
        
    # test_number = 165 
    # for i in range(test_number):
    #     test_case = []
    #     for j in range(len(attribute_limits)):
    #         repeat = 1
    #         rotate = 1
    #         for k in range(j):
    #             repeat *= int(attribute_limits[k][1])
    #         for k in range(j+1):
    #             rotate *= int(attribute_limits[k][1])
    #         attr = round(((int(attribute_limits[j][3]) - int(attribute_limits[j][2])) / int(attribute_limits[j][1])) * (((i % rotate))//repeat) + 0.0000005) + int(attribute_limits[j][2])
    #         if i == (144 / 2) + 1:
    #             pass
    #         test_case.append(attr)
    #     test_cases.append(test_case)  
    
    attribute_limits = np.array([['citizenship', 2, 0, 1],
                        ['state', 2, 0, 1],
                        ['age', 3, 1, 100],
                        ['sex', 2, 0, 1],
                        ['region', 4, 0, 6],
                        ['income_class', 3, 0, 199],
                        ['dependents_number', 5, 0, 4],
                        ['marital_status', 2, 0, 1]])
    
    # insert back deleted attributes with value '1'        
    z = np.ones((len(test_cases), 3), dtype=np.dtype('U20'))
    expanded_test_cases = np.append(test_cases, z, axis=1)

    for i in range(5):
        x, = np.argwhere(attribute_limits == test_cases[0][i])
        attribute_limits = np.delete(attribute_limits,x[0],0)

    expanded_test_cases[0][5] = str(attribute_limits[0][0])
    expanded_test_cases[0][6] = str(attribute_limits[1][0])
    expanded_test_cases[0][7] = str(attribute_limits[2][0])
    
    np.savetxt('test suite.txt', expanded_test_cases, fmt='%s')


def callfunc():
    filename = 'credit'
    funcname = 'creditApproval'
    # funcname = 'mutatedCreditApproval1'
    attributes = {'age': 18, 'region': 6}
    exec(f"from {filename} import {funcname}\nprint({funcname}(**attributes))")



def main():
    # xor()
    # digits()
    # quadratic()
    # clothesTest()
    # clothesBuild()
    # i = 500
    # while i < 10000:
    #     clothesPredict(i)
    #     i += 1
    
    # buildCreditNN(10)
    # predictCreditNN()
    # inputPredictCreditNN()
    # with cProfile.Profile() as pr:
        # testPredictCredit()
    # generateTestCases(rankAttrCredit())
    # testCredit()
    
    application()
    
    # callfunc()
    
    # saveTestCases()
    # print(rankAttrCredit())
    # print(generateTestCases(rankAttrCredit()))
    
    # stats = pstats.Stats(pr)
    # stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    # stats.dump_stats(filename='needs_profiling.prof')
    
if __name__ == '__main__':
    # This code won't run if this file is imported.
    main()
    