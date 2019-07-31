from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def linear():
    """
    线性回归
    :return: None
    """
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    # print(lb)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    print(lr.coef_)

    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("lr预测值：", y_lr_predict)
    print("lr均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))

    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)

    print(sgd.coef_)

    y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
    print("sgd预测值", y_sgd_predict)
    print("sgd均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print(rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print("rd预测值", y_rd_predict)
    print("rd均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))

    return None


if __name__ == '__main__':
    linear()
