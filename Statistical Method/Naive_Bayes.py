#!usr/bin/env python3.6
# encoding = utf - 8

import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

# 定义函数plot_classifier
def plot_classifiler(classifiler, X, y):
    # 定义图形的取值范围
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    # 这里1.0为余量（buffer）

    # 设置网格数据的步长
    step_size = 0.001

    # 定义网格
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # 计算分类器输出结果
    mesh_output = classifiler.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # 数组维度变形
    mesh_output = mesh_output.reshape(x_values.shape)

    # 用彩图画出分类结果
    plt.figure()

    # 选择配色方案
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # 训练数据点画到图上
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidths=1, cmap=plt.cm.Paired)

    # 设置图形取值范围
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # 设置X轴与Y轴
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))

    plt.show()

# 加载数据文件data_multivar.txt
input_file = '../Data/data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

# 创建一个朴素贝叶斯分类器 GaussianNB正态分布朴素贝叶斯模型
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)

# 计算分类器的准确性
accuracy = 100.0 * (y == y_pred).sum()/X.shape[0]
print("Accuracy of the classifier = ", round(accuracy, 2), "%")

# 画出数据点和边界
plot_classifiler(classifier_gaussiannb, X, y)

# ---*将数据集分割成训练集和测试集*---
# 在0.2以后版本将用train_test_split代替cross_validation
#from sklearn import cross_validation
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

#classifier_gaussiannb_new = GaussianNB()
#classifier_gaussiannb_new.fit(X_train, y_train)

# 用分类器对测试数据进行测试
#y_test_pred = classifier_gaussiannb_new.predict(X_test)

# 计算分类器准确定
#accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
#print("Accuracy of the classifier = ", round(accuracy, 2), "%")

# 画出测试数据点及边界
#plot_classifiler(classifier_gaussiannb_new, X_test, y_test)

# ---*交叉验证检验模型准确性*---
num_validations = 5
accuracy = cross_val_score(classifier_gaussiannb, X, y, scoring='accuracy', cv=num_validations)
print("Accuracy: " + str(round(100 * accuracy.mean(), 2)) + "%")

# 计算精度、召回率和F1得分
f1 = cross_val_score(classifier_gaussiannb, X, y, scoring='f1_weighted', cv=num_validations)
print("F1: " + str(round(100 * f1.mean(), 2)) + "%")

precision = cross_val_score(classifier_gaussiannb, X, y, scoring='precision_weighted', cv=num_validations)
print("Precision: " + str(round(100 * precision.mean(), 2)) + "%")

recall = cross_val_score(classifier_gaussiannb, X, y, scoring='recall_weighted', cv=num_validations)
print("Recall: " + str(round(100 * recall.mean())) + "%")