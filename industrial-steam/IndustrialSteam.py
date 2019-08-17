import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


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
    loss = mean_squared_error(y_test, y_predict)
    print("线性回归均方误差为：", loss)

    return linearRegression.predict(data_test), loss


def rfFunction(x_train, x_test, y_train, y_test, data_test):
    """
    随机森林
    :return: 预测结果，测试损失
    """
    rf = RandomForestRegressor(n_estimators=50)
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)
    loss = mean_squared_error(y_test, y_predict)
    print("随机森林均方误差为：", loss)
    return rf.predict(data_test), loss


def outFile(path, predict):
    # 输出结果
    with open(path, "w+") as f:
        for i in predict:
            f.write(str(i) + "\n")


if __name__ == "__main__":
    # 导入数据
    data = pd.read_csv("./data/zhengqi_train.txt", sep="\t")
    data_test = pd.read_csv("./data/zhengqi_test.txt", sep="\t")
    # 数据分为训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data['target'], test_size=0.25)

    # 1. lr回归
    lr_predict, lr_loss = lrFuntion(x_train, x_test, y_train, y_test, data_test)
    # 2. rf随机森林
    rf_predict, rf_loss = rfFunction(x_train, x_test, y_train, y_test, data_test)

    outFile("./output/lrOutput.txt", lr_predict)
    outFile("./output/rfOutput.txt", rf_predict)
    print("lr,和rf 的均方误差：",mean_squared_error(lr_predict,rf_predict))
