import pandas as pd
from sklearn.model_selection import train_test_split


# 导入数据
data = pd.read_csv("./data/happiness_train_abbr.csv")

# 特征工程


# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 2:], data["happiness"], test_size=0.25)

# 查看数据是缺失程度
x_train.info(verbose=True,null_counts=True)

#查看label分布
print(y_train.value_counts())