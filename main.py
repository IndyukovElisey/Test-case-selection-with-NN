from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from random import random, randint

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("TensorFlow version:", tf.__version__)
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


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


def creditApproval(citizenship, state, age, sex, region, income_class, dependents_number, marital_status):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 1000 * income_class

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
                credit_limit = 1000 + 800 * income_class
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


def mutatedCreditApproval(citizenship, state, age, sex, region, income_class, dependents_number, marital_status):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 0:
                credit_limit = 5000 + 1000 * income_class

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
                credit_limit = 1000 + 800 * income_class
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
    
    for i in range(5000):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = randint(0, 6)
        income_class = randint(0, 3)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 3)
        dependents_number = normalize(dependents_number, 0, 4)
        credit_limit = normalize(credit_limit, 0, 18000)
        if credit_limit != 0:
            credit_limit = int(credit_limit // 0.1) + 1

        if i < 1000:
            train_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
            train_output.append(credit_limit)
        else:
            test_input.append([citizenship, state, age, sex, region, income_class, dependents_number, marital_status])
            test_output.append(credit_limit)

    # print(attributes)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(output_class_number+1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_input, train_output, epochs=300)
    model.save('saved_model/credit_model.h5')

    model.summary()

    test_loss, test_acc = model.evaluate(test_input, test_output, verbose=2)

    print('\nTest accuracy:', test_acc)


def predictCreditNN():
    test_input = []
    test_output = []

    for i in range(100):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = randint(0, 6)
        income_class = randint(0, 3)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 3)
        dependents_number = normalize(dependents_number, 0, 4)
        credit_limit = normalize(credit_limit, 0, 18000)
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
    income_class = normalize(income_class, 0, 3)
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


def testCredit():
    test_input = []
    test_number = 100

    for i in  range(test_number):
        citizenship = randint(0, 1)
        state = randint(0, 1)
        age = randint(1, 100)
        sex = randint(0, 1)
        region = 2
        income_class = randint(0, 3)
        dependents_number = randint(0, 4)
        marital_status = randint(0, 1)

        age = normalize(age, 1, 100)
        region = normalize(region, 0, 6)
        income_class = normalize(income_class, 0, 3)
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
        income_class = denormalize(test_input[i][5],0,3)
        dependents_number = denormalize(test_input[i][6],0,4)
        marital_status = test_input[i][7]
        credit_approved, credit_limit = creditApproval(citizenship, state, age, sex, region, income_class,
                                                       dependents_number, marital_status)
        credit_limit = normalize(credit_limit, 0, 18000)
        if credit_limit != 0:
            credit_limit = int(credit_limit // 0.1) + 1

        credit_approved2, credit_limit2 = mutatedCreditApproval(citizenship, state, age, sex, region, income_class,
                                                                dependents_number, marital_status)
        credit_limit2 = normalize(credit_limit2, 0, 18000)
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


# xor()
# digits()
# quadratic()
# clothesTest()
# clothesBuild()
# i = 500
# while i < 10000:
#     clothesPredict(i)
#     i += 1
# buildCreditNN(20)
# predictCreditNN()
# inputPredictCreditNN()
testCredit()
# print(np.tanh(0), np.tanh(1), np.tanh(100), np.tanh(1000), np.tanh(10000), np.tanh(100000))
