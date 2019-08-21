import numpy as np
import matplotlib.pyplot as plt

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

arr_x = np.array([1.7, 2, 3, 4.5, 5.3, 6, 7, 8, 9, 10.5])

print("准备特征相关性绘图：\n", arr_x)
# 其中alpha表示透明度，越小越透明；width表示柱状图的宽度；color表示柱状图的颜色
plt.bar(np.arange(10), arr_x, alpha=0.7, width=0.3, color='blue')
plt.title("特征相关性")
plt.show()
