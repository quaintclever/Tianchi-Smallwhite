import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

def lr_function(x_train, x_test, y_train, y_test, data_test):
    """
    线性回归预测工业蒸汽
    :return:
    """
    # 构造线性回归
    lr = LinearRegression()
    # 数据训练
    lr.fit(x_train, y_train)
    # 数据预测值
    y_predict = lr.predict(x_test)
    # 均方误差
    loss = mean_squared_error(y_test, y_predict)
    print("线性回归均方误差为：", loss)

    return lr.predict(data_test), loss


def rf_function(x_train, x_test, y_train, y_test, data_test):
    """
    随机森林
    :return: 预测结果，测试损失
    """

    # # 网格搜索参数
    # param = {
    #     "n_estimators": [375, 400, 425],
    #     "max_depth": [12, 14, 16, 18, 20]
    # }
    # # 随机森林 估计器
    # rf = RandomForestRegressor()
    # # 定义网格搜索，5折 取最小 mse
    # gcv = GridSearchCV(estimator=rf, param_grid=param, cv=5, scoring="neg_mean_squared_error")
    # gcv.fit(x_train, y_train)
    # # 取参数最好的估计器
    # best_rf = gcv.best_estimator_
    # # 打印出最好的参数
    # best_param = gcv.best_params_
    # print("调好的参数为：", best_param)
    # y_predict = best_rf.predict(x_test)
    # loss = mean_squared_error(y_test, y_predict)
    # print("随机森林均方误差为：", loss)
    # return best_rf.predict(data_test), loss

    # 随机森林 估计器
    rf = RandomForestRegressor(n_estimators=400,max_depth=18)
    rf.fit(x_train, y_train)
    y_predict = rf.predict(x_test)
    loss = mean_squared_error(y_test, y_predict)
    print("随机森林均方误差为：", loss)
    rf_fi = rf.feature_importances_
    print("准备特征相关性绘图：\n", rf_fi)
    # 其中alpha表示透明度，越小越透明；width表示柱状图的宽度；color表示柱状图的颜色
    plt.bar(np.arange(38), rf_fi, alpha=0.7, width=0.3, color='blue')
    plt.title("特征相关性")
    plt.show()
    return rf.predict(data_test), loss


def out_file(path, predict):
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
    lr_predict, lr_loss = lr_function(x_train, x_test, y_train, y_test, data_test)
    # 2. rf随机森林
    rf_predict, rf_loss = rf_function(x_train, x_test, y_train, y_test, data_test)

    out_file("./output/lrOutput.txt", lr_predict)
    out_file("./output/rfOutput.txt", rf_predict)
    print("lr,和rf 的均方误差：", mean_squared_error(lr_predict, rf_predict))
