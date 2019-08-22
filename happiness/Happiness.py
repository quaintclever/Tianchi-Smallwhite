import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

# 导入数据
train = pd.read_csv("./data/happiness_train_abbr.csv")
test = pd.read_csv("./data/happiness_test_abbr.csv")


# 特征工程
def fill_na_data(data):
    """
    填充空数据 ， 特征工程
    :param data: 数据
    :return:
    """
    # 去掉四个缺失值很多的
    data = data.drop(["work_status"], axis=1)
    data = data.drop(["work_yr"], axis=1)
    data = data.drop(["work_type"], axis=1)
    data = data.drop(["work_manage"], axis=1)
    # 把所有的 -8 都换成 np.nan
    data = data.applymap(lambda x: np.nan if x == -8 else x)

    data["nationality"] = data["nationality"].fillna(8)
    data["edu"] = data["edu"].fillna(14)
    data["political"] = data["political"].fillna(1)
    data["health"] = data["health"].fillna(3)
    data["health_problem"] = data["health_problem"].fillna(4)
    data["depression"] = data["depression"].fillna(4)
    data["socialize"] = data["socialize"].fillna(2)
    data["relax"] = data["relax"].fillna(4)
    data["learn"] = data["learn"].fillna(1)
    data["equity"] = data["equity"].fillna(3)
    data["class"] = data["class"].fillna(5)
    data["family_status"] = data["family_status"].fillna(3)
    data["status_peer"] = data["status_peer"].fillna(2)
    data["status_3_before"] = data["status_3_before"].fillna(2)
    data["view"] = data["view"].fillna(3)
    data["inc_ability"] = data["inc_ability"].fillna(2)
    # 宗教信仰 0 否 1 是 用2 填补 nan
    data["religion"] = data["religion"].fillna(2)
    # 1 = 从来没有参加过; 2 = 一年不到1次; 3 = 一年大概1到2次; 4 = 一年几次; 5 = 大概一月1次;
    # 6 = 一月2到3次; 7 = 差不多每周都有; 8 = 每周都有; 9 = 一周几次;
    data["religion_freq"] = data["religion_freq"].fillna(0)
    # 1 = 有; 2 = 没有;
    data["car"] = data["car"].fillna(0)
    # 用平均值 填充 家庭年收入的 nan
    data["family_income"] = data["family_income"].fillna(data["family_income"].mean())

    # 个人年收入 income 分组处理
    def income_cut(x):
        if x < 0:
            return 0
        elif 0 <= x < 1200:
            return 1
        elif 1200 <= x < 10000:
            return 2
        elif 10000 <= x < 24000:
            return 3
        elif 24000 <= x < 40000:
            return 4
        elif 40000 <= x:
            return 5

    data["income_cut"] = data["income"].map(income_cut)
    # 删除原来的 个人年收入
    data = data.drop(["income"], axis=1)

    # 处理时间特征
    data['survey_time'] = pd.to_datetime(data['survey_time'], format='%Y-%m-%d %H:%M:%S')
    data["survey_year"] = data["survey_time"].dt.year

    # 获取年龄
    data["age"] = data["survey_year"] - data["birth"]
    # 删除原来的 调查时间
    data = data.drop(["survey_time"], axis=1)

    # 对年龄进行分组
    def age_cut(x):
        if 10 <= x < 20:
            return 0
        elif 20 <= x < 30:
            return 1
        elif 30 <= x < 40:
            return 2
        elif 40 <= x < 50:
            return 3
        elif 50 <= x < 60:
            return 4
        elif 60 <= x < 70:
            return 5
        elif 70 <= x < 80:
            return 6
        elif 80 <= x < 90:
            return 7
        elif 90 <= x < 100:
            return 8

    data["age_cut"] = data["age"].map(age_cut)
    # 删除原来的 年龄
    data = data.drop(["age"], axis=1)
    return data


# 特征值数据填充
train["happiness"] = train["happiness"].fillna(3)
# 训练集特征处理
train = fill_na_data(train)
# 测试集特征处理
test = fill_na_data(test)

# 分割数据集 id happiness 不参与训练
x_train, x_test, y_train, y_test = train_test_split(train.iloc[:, 2:], train["happiness"], test_size=0.25)

# 模型训练
rfc = RandomForestClassifier(n_estimators=400, max_depth=20)
rfc.fit(x_train, y_train)

# 预测 ，计算 mse
y_predict = rfc.predict(x_test)
loss = mean_squared_error(y_test, y_predict)
print("mse ：", loss)

# 预测test
test_predict = rfc.predict(test.iloc[:, 1:])

# 输出数据
# with open("./output/rf_happiness_submit.csv", "w+") as f:
#     f.write("id,happiness\n")
#     for i,h in enumerate(test_predict):
#         f.write(str(i+8001) +","+str(int(h))+ "\n")
