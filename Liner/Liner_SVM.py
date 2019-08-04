#!usr/bin/env python3.6
# encoding = utf - 8

import numpy as np
import matplotlib.pyplot as plt


# 加载输入数据
# 正常数据集
# input_file = 'data_multivar.txt'
# 数据不平衡数据集 data_multivar_imbalance
input_file = '../data/data_multivar_imbalance.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])

X = np.array(X)
y = np.array(y)

# 将数据分成类
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])

# 画图像
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolor='black', edgecolors='black', marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolor='None', edgecolors='black', marker='s')
plt.title('Input data')
plt.show()

# 分格数据集并用SVM训练模型
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# 用核函数初始化一个SVM对象

# 线性SVM分类器
params = {'kernel': 'linear'}

# 三次多项式核函数
# params = {'kernel': 'poly', 'degree': 3}

# 径向基函数
# params = {'kernel': 'rbf'}

# 权重核函数
# params = {'kernel': 'linear', 'class_weight': 'auto'}

classifier = SVC(**params)

# 训练分类器
classifier.fit(X_train, y_train)

# Start 分类画图函数 #
def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
    # define ranges to plot the figure
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # Set the title
    plt.title(title)

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks(())
    plt.yticks(())

    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            # Full documentation of the function available here:
            # http://matplotlib.org/api/text_api.html#matplotlib.text.Annotation
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy=(x, y), xytext=(-15, 15),
                textcoords='offset points',
                horizontalalignment='right',
                verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.6', fc='white', alpha=0.8),
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0'))
# End 分类画图函数 #


# 分类器执行
plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()

# 分类器对测试数据执行
y_test_pred = classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()

# 计算训练数据集的准确性
from sklearn.metrics import classification_report

target_names = ['Class-' + str(int(i)) for i in set(y)]
#print("\n" + "#" * 30)
#print("\nClassifier performance on training dataset\n")
#print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))
#print("\n" + "#" * 30)

# 看分类器为测试数据生成的报告
print("#" * 30)
print("\nClassification report on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#" * 30 + "\n")

plt.show()
