from __future__ import print_function
from unicodedata import name

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
# from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from datetime import datetime
from random import random, randint
from collections import OrderedDict
import cProfile
import pstats

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import *

import sys
import io
import ast
import inspect

np.set_printoptions(threshold=sys.maxsize)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# print("TensorFlow version:", tf.__version__)
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow **IS** using the GPU")
else:
    print("TensorFlow **IS NOT** using the GPU")


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        loadUi("NN.ui", self)
        self.btn_save_program.clicked.connect(self.saveProgram)
        self.btn_save_attributes.clicked.connect(self.saveAttributes)
        self.btn_createNN.clicked.connect(self.buildNN)
        self.btn_sort.clicked.connect(self.rankAttributes)
        self.btn_generate.clicked.connect(self.generateTestCases)
        self.btn_test.clicked.connect(self.testCredit)
        self.btn_page_setup.clicked.connect(
            lambda: self.verticalStackedWidget.setCurrentWidget(self.page_setup))
        self.btn_page1.clicked.connect(
            lambda: self.verticalStackedWidget.setCurrentWidget(self.page_1))
        self.btn_page2.clicked.connect(
            lambda: self.verticalStackedWidget.setCurrentWidget(self.page_2))
        self.btn_page3.clicked.connect(
            lambda: self.verticalStackedWidget.setCurrentWidget(self.page_3))
        self.btn_page4.clicked.connect(
            lambda: self.verticalStackedWidget.setCurrentWidget(self.page_4))
        self.action1.triggered.connect(self.aboutDialog)
        self.resize(1280, 720)
        self.show()

        header = self.attribute_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)

        header = self.range_table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        self.filename = 'credit'
        self.funcname = 'creditApproval'
        self.mutated_funcname = ['mutatedCreditApproval1',
                                 'mutatedCreditApproval2',
                                 'mutatedCreditApproval3',
                                 'mutatedCreditApproval4',
                                 'mutatedCreditApproval5',
                                 'mutatedCreditApproval6',
                                 'mutatedCreditApproval7',
                                 'mutatedCreditApproval8',
                                 'mutatedCreditApproval9',
                                 'mutatedCreditApproval10']
        self.attributes = []
        self.output = ['credit_approved', 'credit_limit']

        self.attribute_limits = [[0, 1],
                                 [0, 1],
                                 [1, 100],
                                 [0, 1],
                                 [0, 6],
                                 [0, 199],
                                 [0, 4],
                                 [0, 1]]
        self.output_limits = [0, 20000]
        self.attribute_ranges = []
        self.ranked_attributes = []
        self.test_suite_path = 'test suite.txt'
        self.NNpath = 'saved_model/credit_model1.h5'

    def buildNN(self):
        try:
            train_input = []
            train_output = []
            test_input = []
            test_output = []

            epochs_number = self.epochs.value()
            train_number = self.train.value()
            test_number = self.train.value()
            output_class_number = self.range.value()
            self.NNpath = f"saved_model/{self.NN_name1.text()}.h5"
            out = io.StringIO()

            sys.stdout = out

            for i in range(train_number + test_number):
                attributes = {}
                for j in range(len(self.attributes)):
                    attributes[self.attributes[j]] = randint(
                        self.attribute_limits[j][0], self.attribute_limits[j][1])

                # citizenship = randint(
                #     self.attribute_limits1[0][0], self.attribute_limits1[0][1])
                # state = randint(
                #     self.attribute_limits1[1][0], self.attribute_limits1[1][1])
                # age = randint(
                #     self.attribute_limits1[2][0], self.attribute_limits1[2][1])
                # sex = randint(
                #     self.attribute_limits1[3][0], self.attribute_limits1[3][1])
                # region = randint(
                #     self.attribute_limits1[4][0], self.attribute_limits1[4][1])
                # income_class = randint(
                #     self.attribute_limits1[5][0], self.attribute_limits1[5][1])
                # dependents_number = randint(
                #     self.attribute_limits1[6][0], self.attribute_limits1[6][1])
                # marital_status = randint(
                #     self.attribute_limits1[7][0], self.attribute_limits1[7][1])

                # attributes = {'citizenship': citizenship, 'state': state, 'age': age, 'sex': sex, 'region': region,
                #               'income_class': income_class, 'dependents_number': dependents_number, 'marital_status': marital_status}
                exec(
                    f"from {self.filename} import {self.funcname}\nexec_output={self.funcname}(**attributes)")
                output = locals()['exec_output']
                if isinstance(output, tuple):
                    output = output[1]

                for j in range(len(self.attributes)):
                    attributes[self.attributes[j]] = normalize(
                        attributes[self.attributes[j]], self.attribute_limits[j][0], self.attribute_limits[j][1])
                # age = normalize(
                #     age, self.attribute_limits1[2][0], self.attribute_limits1[2][1])
                # region = normalize(
                #     region, self.attribute_limits1[4][0], self.attribute_limits1[4][1])
                # income_class = normalize(
                #     income_class, self.attribute_limits1[5][0], self.attribute_limits1[5][1])
                # dependents_number = normalize(
                #     dependents_number, self.attribute_limits1[6][0], self.attribute_limits1[6][1])
                if output > self.output_limits[1]:
                    raise Exception(
                        "ОШИБКА: выходное значение превысило верхний предел")
                if output < self.output_limits[0]:
                    raise Exception(
                        "ОШИБКА: выходное значение меньше нижнего предела")

                output = normalize(
                    output, self.output_limits[0], self.output_limits[1])
                if output != 0:
                    output = int(output // (1 / output_class_number)) + 1

                if i < train_number:
                    train_input.append(list(attributes.values()))
                    train_output.append(output)
                else:
                    test_input.append(list(attributes.values()))
                    test_output.append(output)

            # print(attributes)

            # model = tf.keras.Sequential([
            #     tf.keras.layers.Dense(10, input_dim=len(
            #         self.attributes), activation='relu'),
            #     tf.keras.layers.Dense(10, activation='relu'),
            #     tf.keras.layers.Dense(output_class_number+1)
            # ])

            self.statusBar().showMessage("Получены тренировочные наборы")

            model = tf.keras.Sequential([
                tf.keras.layers.Dense(30, input_dim=len(
                    self.attributes), activation='relu'),
                tf.keras.layers.Dense(output_class_number+1)
            ])

            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])

            early_stopping_monitor = EarlyStopping(
                monitor="accuracy", patience=500)

            model.fit(train_input, train_output, epochs=epochs_number,
                      callbacks=[early_stopping_monitor])
            model.save(self.NNpath)
            model.summary()

            self.statusBar().showMessage("Создана нейронная сеть")

            self.result_createNN.setText(out.getvalue())

            test_loss, test_acc = model.evaluate(
                test_input, test_output, verbose=2)
            self.result_accurasy.setText(
                f'Точность нейронной сети на тестовом наборе: {test_acc}')

            sys.stdout = sys.__stdout__
        except Exception as e:
            self.statusBar().showMessage(str(e))
            return

    def rankAttributes(self):
        try:
            self.NNpath = f"saved_model/{self.NN_name2.text()}.h5"
            # rank NN input attributes based on weight values
            new_model = tf.keras.models.load_model(self.NNpath)
            # new_model.summary()

            # get weights
            weights0 = np.array(new_model.layers[0].get_weights()[0])
            weights1 = np.array(new_model.layers[1].get_weights()[0])
            # weights2 = new_model.layers[2].get_weights()[0]

            weight_rank = np.empty((0, 3), dtype=float)
            while True:
                minp = 99999
                min_iw = -1
                min_iw1 = -1
                for iw, w in enumerate(weights0):
                    for iw1, w1 in enumerate(w):
                        max_v = 0
                        for v in weights1[iw1]:
                            if max_v < abs(w1 * v):
                                max_v = abs(w1 * v)
                        if minp > max_v:
                            minp = max_v
                            min_iw = iw
                            min_iw1 = iw1
                if (min_iw == -1 and min_iw1 == -1) or weights0[min_iw][min_iw1] == 99999:
                    break
                weight_rank = np.append(
                    weight_rank, [[min_iw, min_iw1, minp]], axis=0)
                weights0[min_iw][min_iw1] = 99999
                pass

            self.ranked_attributes = []
            # iteratively delete weights, then rank attributes based on deletion order
            while len(weight_rank) > 0:
                if [i[0] for i in weight_rank].count(weight_rank[0][0]) == 1:
                    self.ranked_attributes.append(int(weight_rank[0][0]))
                weight_rank = weight_rank[1:]

            # swap attribute ids to names
            for i in range(len(self.ranked_attributes)):
                # print(attributes[i])
                self.ranked_attributes[i] = self.attributes[self.ranked_attributes[i]]

            # reverse list, so that most important attributes are first
            self.ranked_attributes = list(reversed(self.ranked_attributes))

            # weights0 = new_model.layers[0].get_weights()[0]
            # weights1 = new_model.layers[1].get_weights()[0]
            # weight_rank1 = []

            # # sort weights
            # while weights0.size > len(weight_rank1):
            #     min = 99999
            #     xmin = 0
            #     ymin = 0
            #     for ix, iy in np.ndindex(weights0.shape):
            #         if abs(weights0[ix][iy]) < abs(min) and (weights0[ix][iy] != weight_rank1).all():
            #             min = weights0[ix][iy]
            #             xmin = ix
            #             ymin = iy
            #     weight_rank1.append([xmin, ymin, min])

            # self.ranked_attributes = []
            # # iteratively delete weights, then rank attributes based on deletion order
            # while len(weight_rank1) > 0:
            #     if [i[0] for i in weight_rank1].count(weight_rank1[0][0]) == 1:
            #         self.ranked_attributes.append(weight_rank1[0][0])
            #     weight_rank1.pop(0)

            # # swap attribute ids to names
            # for i in range(len(self.ranked_attributes)):
            #     # print(attributes[i])
            #     self.ranked_attributes[i] = self.attributes[self.ranked_attributes[i]]

            # # reverse list, so that most important attributes are first
            # self.ranked_attributes = list(reversed(self.ranked_attributes))

            self.ranked_table.setRowCount(len(self.ranked_attributes))
            for i, temp_attr in enumerate(self.ranked_attributes):
                self.ranked_table.setItem(i, 0, QTableWidgetItem(temp_attr))

            header = self.ranked_table.horizontalHeader()
            header.setSectionResizeMode(
                0, QtWidgets.QHeaderView.ResizeToContents)
            self.statusBar().showMessage("Получен ранжированный список атрибутов")
            # self.result_sort.setText(f'{self.ranked_attributes}')
        except Exception as e:
            self.statusBar().showMessage(str(e))

    def generateTestCases(self):
        try:
            # generate test cases based on ranked list of input attributes

            # rank_list = ast.literal_eval(self.result_sort.toPlainText())
            test_cases = []
            # attribute_limits = self.attribute_limits
            attribute_limits1 = self.attribute_limits
            test_number = 1

            # read number of attributes' ranges from table
            temp_ranges = []
            for i in range(len(self.attributes)):
                temp_ranges.append(int(self.range_table.item(i, 0).text()))
                if int(self.range_table.item(i, 0).text()) < 1:
                    raise Exception(
                        "ОШИБКА: минимальное количество диапазонов: 1")

            self.attribute_ranges = temp_ranges

            if len(attribute_limits1) == 0:
                raise Exception(
                    'ОШИБКА: Отсутствуют граничные значения атрибутов')
            if len(self.ranked_attributes) == 0:
                raise Exception(
                    'ОШИБКА: Отсутствует ранжированный список атрибутов')

            # put attribute names in first row of test suite
            temp_header = []
            for i in self.ranked_attributes:
                temp_header.append(i)
            test_cases.append(temp_header)

            # count number of test cases based on attributes' number of possible values
            for i in range(len(self.attribute_ranges)):
                test_number *= self.attribute_ranges[i]

            # match number of ranges to ranked attributes
            matched_ranges = []
            for i in self.attributes:
                matched_ranges.append(
                    self.attribute_ranges[self.ranked_attributes.index(i)])
            # self.attribute_ranges = temp_ranges

            # generate test cases
            for i in range(test_number):
                test_case = []
                for j in range(len(attribute_limits1)):
                    repeat = 1
                    rotate = 1
                    for k in range(j):
                        repeat *= self.attribute_ranges[k]
                    for k in range(j+1):
                        rotate *= self.attribute_ranges[k]
                    limits = self.attribute_limits[self.attributes.index(
                        self.ranked_attributes[j])]
                    if self.attribute_ranges[j] == 1:
                        attr = int((limits[1] - limits[0]) / 2)
                    else:
                        left_rand = int(
                            ((limits[1] - limits[0]) / (self.attribute_ranges[j])) * (((i % rotate))//repeat) + limits[0])
                        right_rand = int(((limits[1] - limits[0]) / (self.attribute_ranges[j])) * (
                            ((i % rotate))//repeat + 1) + limits[0])
                        attr = randint(left_rand, right_rand)
                        # attr = round(((limits[1] - limits[0]) / self.attribute_ranges[j]) * (((i % rotate))//repeat) + 0.0000005) + limits[0]
                        # attr = int(((limits[1] - limits[0]) / (self.attribute_ranges[j] - 1)) * (((i % rotate))//repeat) + limits[0])
                    # if i == (144 / 2) + 1:
                    #     pass
                    test_case.append(attr)
                test_cases.append(test_case)

            # show test cases in table
            self.cases_table.setRowCount(len(test_cases)-1)
            self.cases_table.setColumnCount(len(self.attributes))
            a = []
            for i in range(len(self.attributes)):
                a.append(self.ranked_attributes[i])

            self.cases_table.setHorizontalHeaderLabels(a)
            header = self.cases_table.horizontalHeader()

            for i in range(len(test_cases) - 1):
                for j in range(len(self.attributes)):
                    self.cases_table.setItem(
                        i, j, QTableWidgetItem(str(test_cases[i+1][j])))

            for j in range(len(self.attributes)):
                header.setSectionResizeMode(
                    j, QtWidgets.QHeaderView.ResizeToContents)

            np.savetxt(self.test_suite_path, test_cases, fmt='%s')
            self.statusBar().showMessage("Тестовые кейсы сохранены")
            #     rowPosition = self.cases_table.rowCount()
            #     self.cases_table.insertRow(rowPosition)
            #     item = QTableWidgetItem(i)
            #     item.setFlags(QtCore.Qt.ItemIsEnabled)
            #     self.cases_table.setItem(rowPosition , 0, item)
            #     self.cases_table.setItem(rowPosition , 1, QTableWidgetItem("0"))
            #     self.cases_table.setItem(rowPosition , 2, QTableWidgetItem("0"))

            # # delete 3 least important attributes
            # for i in range(len(self.ranked_attributes)):
            #     if i < 3:
            #         x, = np.argwhere(attribute_limits == self.ranked_attributes[i])
            #         attribute_limits = np.delete(attribute_limits, x[0], 0)

            # # attribute names in first row of test suite
            # test_cases.append([attribute_limits[0][0], attribute_limits[1][0],
            #                    attribute_limits[2][0], attribute_limits[3][0],
            #                    attribute_limits[4][0]])

            # generate test cases

            # for i in range(test_number):
            #     test_case = []
            #     for j in attribute_limits:
            #         attr = randint(int(j[2]), int(j[3]))
            #         test_case.append(attr)
            #     test_cases.append(test_case)

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

            # attribute_limits = self.attribute_limits

            # insert back deleted attributes with value '1'
            # z = np.zeros((len(test_cases), 3), dtype=int)
            # z = z.astype(np.dtype('U20'))
            # expanded_test_cases = np.append(test_cases, z, axis=1)

            # for i in range(5):
            #     x, = np.argwhere(attribute_limits == test_cases[0][i])
            #     attribute_limits = np.delete(attribute_limits, x[0], 0)

            # expanded_test_cases[0][5] = str(attribute_limits[0][0])
            # expanded_test_cases[0][6] = str(attribute_limits[1][0])
            # expanded_test_cases[0][7] = str(attribute_limits[2][0])

            # np.savetxt(self.test_suite_path, expanded_test_cases, fmt='%s')
            # self.result_generate.setText(str(expanded_test_cases))

            # np.savetxt(self.test_suite_path, test_cases, fmt='%s')
            # self.result_generate.setText(str(np.array(test_cases)))
        except ValueError:
            self.statusBar().showMessage(
                "ОШИБКА: в количестве диапазонов может быть только целое число")
            return
        except Exception as e:
            self.statusBar().showMessage(str(e))
            return

    def testCredit(self):
        try:
            # compares mutated and original version with given test cases
            test_cases = np.loadtxt(
                self.test_suite_path, dtype=np.dtype('U20'))

            error_found = 0
            test_number = 0
            cases_found_error = ''

            # attribute_limits = np.array([['citizenship', 2, 0, 1],
            #                     ['state', 2, 0, 1],
            #                     ['age', 3, 1, 100],
            #                     ['sex', 2, 0, 1],
            #                     ['region', 7, 0, 6],
            #                     ['income_class', 4, 0, 199],
            #                     ['dependents_number', 5, 0, 4],
            #                     ['marital_status', 2, 0, 1]])

            for i in range(1, len(test_cases)):
                attributes = {}
                test_number += 1

                for j in range(8):
                    attributes[test_cases[0][j]] = int(test_cases[i][j])

                exec(
                    f"from {self.filename} import {self.funcname}\nexec_output={self.funcname}(**attributes)")

                original_result1, original_result2 = locals()['exec_output']

                exec(
                    f"from {self.filename} import {self.mutated_funcname[self.error.currentIndex()]}\nexec_output={self.mutated_funcname[self.error.currentIndex()]}(**attributes)")

                mutated_result1, mutated_result2 = locals()['exec_output']

                # if self.error.currentIndex() == 0:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval1(**attributes)
                # elif self.error.currentIndex() == 1:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval2(**attributes)
                # elif self.error.currentIndex() == 2:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval3(**attributes)
                # elif self.error.currentIndex() == 3:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval4(**attributes)
                # elif self.error.currentIndex() == 4:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval5(**attributes)
                # elif self.error.currentIndex() == 5:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval6(**attributes)
                # elif self.error.currentIndex() == 6:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval7(**attributes)
                # elif self.error.currentIndex() == 7:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval8(**attributes)
                # elif self.error.currentIndex() == 8:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval9(**attributes)
                # else:
                #     mutated_result1, mutated_result2 = mutatedCreditApproval10(**attributes)

                if original_result2 != mutated_result2:
                    cases_found_error += f"Кейс №{i} обнаружил ошибку\n"
                    error_found += 1
            # print(f"{test_number} tests, {error_found} errors, error rate: {error_found/test_number}")
            self.result_test.setText(cases_found_error)
            self.test_number.setText(str(test_number))
            self.test_error_found.setText(str(error_found))
            self.test_error_rate.setText(f'{error_found/test_number*100}%')

            self.statusBar().showMessage("Тестирование выполнено")
        except Exception as e:
            self.statusBar().showMessage(str(e))

    def aboutDialog(self):
        dlg = QMessageBox(self)
        dlg.setWindowTitle("О программе")
        dlg.setText(
            "Эта программа предназначена для отбора и генерации тестовых кейсов с применением нейронной сети\nРазработчик: Индюков Е.П.")
        button = dlg.exec()

        if button == QMessageBox.Ok:
            print("OK!")

    def saveProgram(self):
        try:
            self.filename = self.text_filename.text()
            self.funcname = self.text_funcname.text()

            exec(
                f"from {self.filename} import {self.funcname}\nexec_output=get_signature({self.funcname})")
            self.attributes = locals()['exec_output']

            # формирование таблицы с граничными значениями атрибутов
            self.attribute_table.setRowCount(0)
            for i in self.attributes:
                rowPosition = self.attribute_table.rowCount()
                self.attribute_table.insertRow(rowPosition)
                item = QTableWidgetItem(i)
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                self.attribute_table.setItem(rowPosition, 0, item)
                self.attribute_table.setItem(
                    rowPosition, 1, QTableWidgetItem("0"))
                self.attribute_table.setItem(
                    rowPosition, 2, QTableWidgetItem("10"))

            header = self.attribute_table.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            header.setSectionResizeMode(
                1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(
                2, QtWidgets.QHeaderView.ResizeToContents)

            # формирование таблицы с количеством диапазонов атрибутов
            self.range_table.setRowCount(0)
            for i in range(len(self.attributes)):
                rowPosition = self.range_table.rowCount()
                self.range_table.insertRow(rowPosition)
                self.range_table.setItem(rowPosition, 0, QTableWidgetItem(
                    f"{(len(self.attributes) - i - 1) // 3 + 1}"))

            header = self.range_table.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

            self.statusBar().showMessage("Параметры тестируемой программы сохранены")
        except Exception as e:
            self.statusBar().showMessage(str(e))

    def saveAttributes(self):
        try:
            temp_limits = []
            for i in range(len(self.attributes)):
                temp_limits.append([int(self.attribute_table.item(i, 1).text()), int(
                    self.attribute_table.item(i, 2).text())])
                if temp_limits[i][0] > temp_limits[i][1]:
                    raise Exception(
                        "ОШИБКА: Нижний порог не может превышать верхний порог")

            self.attribute_limits = temp_limits
            self.output_limits = [int(self.output_lower_limit.text()), int(
                self.output_upper_limit.text())]
            self.statusBar().showMessage("Параметры аргументов сохранены")
        except ValueError:
            self.statusBar().showMessage(
                "ОШИБКА: в граничных значениях могут быть только целые числа")
        except Exception as e:
            self.statusBar().showMessage(str(e))

    def update(self):
        self.label.adjustSize()


def application():
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.resize(1280, 720)
    sys.exit(app.exec_())


def get_signature(fn):
    # get list of arguments from function
    params = inspect.signature(fn).parameters
    args = []
    kwargs = OrderedDict()
    for p in params.values():
        if p.default is p.empty:
            args.append(p.name)
        else:
            kwargs[p.name] = p.default
    return args + list(kwargs.keys())


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
        # Metrics are different values that you want the model to track while training
        metrics=['binary_accuracy']
    )

    # Our function to take in two numerical inputs and output the relevant boolean
    def cleanPredict(a, b):
        inputTens = tf.constant([[a, b]])
        # model.predict(input) yields a 2d tensor
        return round(model.predict(inputTens)[0][0]) == 1

    # Will yield a random value because model isn't yet trained
    print(cleanPredict(1, 0))

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

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
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

    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
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
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
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
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
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


def creditApproval(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
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


def mutatedCreditApproval1(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
    credit_limit = 0
    credit_approved = 0

    if region == 5:
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


def mutatedCreditApproval2(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
    credit_limit = 0
    credit_approved = 0

    if region == 4 or region == 5:
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


def mutatedCreditApproval3(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age > 18:
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


def mutatedCreditApproval4(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 25:
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


def mutatedCreditApproval5(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
    credit_limit = 0
    credit_approved = 0

    if region == 5 or region == 6:
        credit_limit = 0
    else:
        if age < 18:
            credit_limit = 0
        else:
            if citizenship == 1:
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


def mutatedCreditApproval6(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
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

                if state == 1:
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


def mutatedCreditApproval7(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
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
                    if region == 3:
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


def mutatedCreditApproval8(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
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
                    if region == 1 or region == 2:
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


def mutatedCreditApproval9(citizenship=0, state=0, age=20, sex=0, region=3, income_class=2, dependents_number=1, marital_status=0):
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
                    credit_limit += 5000
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
                credit_limit = 1000 + 2 * income_class
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
    if x == min:
        return 0
    if x == max:
        return 1
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
            train_input.append([citizenship, state, age, sex, region,
                               income_class, dependents_number, marital_status])
            train_output.append(credit_limit)
        else:
            test_input.append([citizenship, state, age, sex, region,
                              income_class, dependents_number, marital_status])
            test_output.append(credit_limit)

    # print(attributes)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_dim=8, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(output_class_number+1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
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

        test_input.append([citizenship, state, age, sex, region,
                          income_class, dependents_number, marital_status])
        test_output.append(credit_limit)

    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    new_model.summary()
    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_input)

    for i in range(100):
        print(np.argmax(predictions[i]),
              test_output[i], test_input[i], predictions[i])


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

    for i in range(test_number):
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

        test_input.append([citizenship, state, age, sex, region,
                          income_class, dependents_number, marital_status])

    new_model = tf.keras.models.load_model('saved_model/credit_model.h5')
    new_model.summary()
    probability_model = tf.keras.Sequential([new_model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_input)

    for i in range(test_number):
        citizenship = test_input[i][0]
        state = test_input[i][1]
        age = denormalize(test_input[i][2], 1, 100)
        sex = test_input[i][3]
        region = denormalize(test_input[i][4], 0, 6)
        income_class = denormalize(test_input[i][5], 0, 199)
        dependents_number = denormalize(test_input[i][6], 0, 4)
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
    attributes = ['citizenship', 'state', 'age', 'sex', 'region',
                  'income_class', 'dependents_number', 'marital_status']
    for i in range(len(input_rank)):
        # print(attributes[i])
        input_rank[i] = attributes[input_rank[i]]
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
            attribute_limits = np.delete(attribute_limits, x[0], 0)

    # attribute names in first row of suite
    test_cases.append([attribute_limits[0][0], attribute_limits[1][0],
                      attribute_limits[2][0], attribute_limits[3][0],
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
        attribute_limits = np.delete(attribute_limits, x[0], 0)

    expanded_test_cases[0][5] = str(attribute_limits[0][0])
    expanded_test_cases[0][6] = str(attribute_limits[1][0])
    expanded_test_cases[0][7] = str(attribute_limits[2][0])

    np.savetxt('test suite.txt', expanded_test_cases, fmt='%s')


def callfunc():
    filename = 'credit'
    funcname = 'creditApproval'
    # funcname = 'mutatedCreditApproval1'
    attributes = {'age': 18, 'region': 6}
    exec(f"from {filename} import {funcname}\nprint({funcname}(**attributes))\nprint(inspect.signature({funcname}))")

    def get_signature(fn):
        params = inspect.signature(fn).parameters
        args = []
        kwargs = OrderedDict()
        for p in params.values():
            if p.default is p.empty:
                args.append(p.name)
            else:
                kwargs[p.name] = p.default
        return args + list(kwargs.keys())

    def fn(a, b, c, d=3, e="abc"):
        pass

    print(get_signature(creditApproval))


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
