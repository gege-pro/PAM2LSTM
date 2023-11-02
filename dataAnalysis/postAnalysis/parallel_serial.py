import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as ticker
from matplotlib.patches import ConnectionPatch

plt.rc('font', family='Times New Roman', size=24)


def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")
    # ax.set_xticks([])
    # ax.set_yticks([])

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)
    """
    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    """

data = pd.read_csv('wy_LSTM_P&S(1).csv')
# 并行预测指标
MAE = mean_absolute_error(data['real'], data['pred'])
RMSE = np.sqrt(mean_squared_error(data['real'], data['pred']))
SS_R = sum((data['real']-data['pred'])**2)
SS_T = sum((data['real']-np.mean(data['real']))**2)
NSE = 1-(float(SS_R))/SS_T

# 串行预测指标
MAE_s = mean_absolute_error(data['real'], data['pred_s'])
RMSE_s = np.sqrt(mean_squared_error(data['real'], data['pred_s']))
SS_R_s = sum((data['real']-data['pred_s'])**2)
NSE_s = 1-(float(SS_R_s))/SS_T

print("Serial:", MAE_s, RMSE_s, NSE_s)
print("Parallel:", MAE, RMSE, NSE)

# 找波峰
Pre_peak = signal.find_peaks(data['pred'], distance=200)
Pre_peak_s = signal.find_peaks(data['pred_s'], distance=200)
Real_peak = signal.find_peaks(data['real'], distance=200)
PreTime, PreTime_s, RealTime = [], [], []
pre_flood, pre_flood_s, real_flood = [], [], []
for i in Pre_peak[0]:
    if data['pred'][i] >= 57*2630/60.58:
        PreTime.append(i)
        pre_flood.append(data['pred'][i])
for i in Pre_peak_s[0]:
    if data['pred_s'][i] >= 57*2630/60.58:
        PreTime_s.append(i)
        pre_flood_s.append(data['pred_s'][i])
for i in Real_peak[0]:
    if data['real'][i] >= 57*2630/60.58:
        RealTime.append(i)
        real_flood.append(data['real'][i])
print("PreTime and RealTime:", PreTime, PreTime_s, RealTime)
print("pre_flood and real_flood:", pre_flood, pre_flood_s, real_flood)
# 画图
fig, ax = plt.subplots(figsize=(10, 8))
# 调整图像留白大小
# fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
for i in range(len(PreTime)):
    # 并行预测值
    ax.plot(PreTime[i], pre_flood[i], 'o', markersize=10, color='dodgerblue')
    ax.text(PreTime[i], pre_flood[i]+0.1, (PreTime[i], int(pre_flood[i])), color='dodgerblue')
    # 真实值
    ax.plot(RealTime[i], real_flood[i], '*', markersize=10, color='r')
    ax.text(RealTime[i], real_flood[i]-0.1, (RealTime[i], int(real_flood[i])), color='r')
    # 串行预测值
    ax.plot(PreTime_s[i], pre_flood_s[i], '^', markersize=10, color='mediumseagreen')
    ax.text(PreTime_s[i], pre_flood_s[i]-0.2, (PreTime_s[i], int(pre_flood_s[i])), color='mediumseagreen')

# plt.axhline(y=57, ls="--", c="r")  # 添加水平直线
x = np.arange(2200, 6200, 1)
y1 = data['pred'][2200:6200]
y2 = data['real'][2200:6200]
y3 = data['pred_s'][2200:6200]
ax.plot(y1, label='parallel network', color='dodgerblue', linewidth=2.1)  # [2200:6200]
ax.plot(y2, label='ground truth', color='r', linestyle='--')
ax.plot(y3, label='serial network', color='mediumseagreen')
ax.grid()
ax.legend()
# plt.title('LSTM-WuYuan')
ax.set_xlabel('Time/h')
ax.set_ylabel('Runoff/ (m$^3$/s)')
ticks = ax.set_xticks([2600, 3936, 4852, 5994])  # 设置刻度
labels = ax.set_xticklabels(['2021/3/28 23:00', '2021/5/23 15:00', '2021/6/30 19:00', '2021/8/17 9:00'], rotation=0)  # 设置刻度标签

# 绘制缩放图
axins1 = ax.inset_axes((0.1, 0.20, 0.2, 0.15))
# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins1.plot(y1, color='dodgerblue',label='trick-1',alpha=0.7)
axins1.plot(y2, color='r', linestyle='--',label='trick-2',alpha=0.7)
axins1.plot(y3, color='mediumseagreen',label='trick-3',alpha=0.7)
# 局部显示并且进行连线
zone_and_linked(ax, axins1, 1720, 1750, x, [y1, y2, y3], 'right')


# 绘制缩放图
axins2 = ax.inset_axes((0.1, 0.40, 0.2, 0.15))
# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins2.plot(y1, color='dodgerblue',label='trick-1',alpha=0.7)
axins2.plot(y2, color='r', linestyle='--',label='trick-2',alpha=0.7)
axins2.plot(y3, color='mediumseagreen',label='trick-3',alpha=0.7)
# 局部显示并且进行连线
zone_and_linked(ax, axins2, 2400, 2430, x, [y1, y2, y3], 'right')

# 绘制缩放图
axins3 = ax.inset_axes((0.1, 0.60, 0.2, 0.15))
# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins3.plot(y1, color='dodgerblue',label='trick-1',alpha=0.7)
axins3.plot(y2, color='r', linestyle='--',label='trick-2',alpha=0.7)
axins3.plot(y3, color='mediumseagreen',label='trick-3',alpha=0.7)
# 局部显示并且进行连线
zone_and_linked(ax, axins3, 2640, 2670, x, [y1, y2, y3], 'right')

# 绘制缩放图
axins4 = ax.inset_axes((0.1, 0.80, 0.2, 0.15))
# 在缩放图中也绘制主图所有内容，然后根据限制横纵坐标来达成局部显示的目的
axins4.plot(y1, color='dodgerblue', label='trick-1',alpha=0.7)
axins4.plot(y2, color='r', linestyle='--', label='trick-2',alpha=0.7)
axins4.plot(y3, color='mediumseagreen', label='trick-3',alpha=0.7)
# 局部显示并且进行连线
zone_and_linked(ax, axins4, 3780, 3810, x, [y1, y2, y3], 'right')

# datawy = pd.read_csv('../../dataSrc/wy_2017_2021.csv', encoding='gbk')
# Rainfall = datawy['SUM'][2200:6200]

plt.show()

