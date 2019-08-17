import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def lrFuntion(x_train, x_test, y_train, y_test, data_test):
    """
    线性回归预测工业蒸汽
    :return:
    """
    # 构造线性回归
    linearRegression = LinearRegression()
    # 数据训练
    linearRegression.fit(x_train, y_train)
    # 数据预测值
    y_predict = linearRegression.predict(x_test)
    # 均方误差
    err = mean_squared_error(y_test, y_predict)
    print("线性回归均方误差为：",err)

    return linearRegression.predict(data_test),err


if __name__ == "__main__":
    # 导入数据
    data = pd.read_csv("./data/zhengqi_train.txt", sep="\t")
    data_test = pd.read_csv("./data/zhengqi_test.txt", sep="\t")
    # 数据分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['target'], test_size=0.25)

    # 1. lr回归
    y_lr_predict, lr_err = lrFuntion(x_train, x_test, y_train, y_test, data_test)

    # 输出结果
    with open("./data/output.txt", "w+") as f:
        for i in y_lr_predict:
            f.write(str(i) + "\n")
