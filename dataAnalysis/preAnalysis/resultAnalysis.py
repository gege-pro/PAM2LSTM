import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits import axisartist
import numpy as np
# from matplotlib import rc
import matplotlib.ticker as ticker
from dataAnalysis.Analysis210803 import corr_coeff
from scipy import signal
# from statsmodels.tsa.stattools import grangercausalitytests


plt.rc('font', family='Times New Roman', size=30)
plt.rcParams['font.sans-serif'] = ['SimHei']
# data = pd.read_csv('../../dataSrc/wy_2017_2021.csv', encoding='gbk')
data = pd.read_csv('../../dataSrc/xx_2010_2018.csv', encoding='gbk')


"""
corr = corr_coeff(data)
np.savetxt('xx_zq.txt', corr)

corr = np.loadtxt('wy_zq.txt')
wy_zwkq = np.loadtxt('wy_zwkq.txt')
wy_sumq = np.loadtxt('wy_sumq.txt')
peak_ind = signal.argrelextrema(corr, np.greater, order=12)
valley_ind = signal.argrelextrema(corr, np.less, order=12)
print(peak_ind, valley_ind)

plt.plot(corr)
plt.plot(signal.argrelextrema(corr, np.greater, order=12)[0], corr[signal.argrelextrema(corr, np.greater, order=12)], '*', markersize=10, label='Maxima')  #极大值点
plt.plot(signal.argrelextrema(corr, np.less, order=12)[0], corr[signal.argrelextrema(corr, np.less, order=12)], 'o', markersize=10, label='Minima')  #极小值点
plt.legend()

# for i in [0, 3]:
    # plt.annotate(s='(peak_ind[0][i], round(corr[peak_ind[0][i]], 4))', xy=(peak_ind[0][i], corr[peak_ind[0][i]]), xytext=(2, 3), weight='bold', color='r',
    #              arrowprops=dict(facecolor='c', shrink=0.05))
    # plt.text(peak_ind[0][i]-18, corr[peak_ind[0][i]]+0.05, (peak_ind[0][i], round(corr[peak_ind[0][i]], 4)))
# for i in range(len(valley_ind[0])):
#     plt.text(valley_ind[0][i], corr[valley_ind[0][i]], (valley_ind[0][i], round(corr[valley_ind[0][i]], 4)))
# plt.text(peak_ind[0][1], corr[peak_ind[0][1]], (peak_ind[0][1], round(corr[peak_ind[0][1]], 4)))
# plt.savefig('R and Z in XX.png')

plt.title('The Autocorrelation of waterlevel in WuYuan')
plt.xlabel('Time interval(h)')
plt.ylabel('Correlation coefficient')
plt.grid()
plt.show()
"""

start = 1070  # 61360
span = 3000
time = data.loc[start:span+start]['TM']
SUM = data.loc[start:span+start]['SUM']
# Z_WK = data.loc[start-40:span+start-40]['Z_WK']
Z = data.loc[start:span+start]['Z']
Q = data.loc[start:span+start]['Q']

# 读spearman系数
corr_wy_sumq = np.loadtxt('xx_sumq.txt')
# corr_wy_zwkq = np.loadtxt('wy_zwkq.txt')
corr_wy_zq = np.loadtxt('xx_zq.txt')
# 获取系数曲线的极大值、极小值
sumq_peak_ind = signal.argrelextrema(corr_wy_sumq, np.greater, order=12)
sumq_valley_ind = signal.argrelextrema(corr_wy_sumq, np.less, order=12)
# zwkq_peak_ind = signal.argrelextrema(corr_wy_zwkq, np.greater, order=12)
# zwkq_valley_ind = signal.argrelextrema(corr_wy_zwkq, np.less, order=12)
zq_peak_ind = signal.argrelextrema(corr_wy_zq, np.greater, order=12)
zq_valley_ind = signal.argrelextrema(corr_wy_zq, np.less, order=12)

Min1 = np.min(Z)
Min2 = np.min(Q)
Max1 = np.max(Z)
Max2 = np.max(Q)

fig = plt.figure()
"""

ax_Q = host_subplot(111, axes_class=axisartist.Axes)
ax_Z = ax_Q.twinx()
ax_ZWK = ax_Q.twinx()
ax_SUM = ax_Q.twinx()

# ax_ZWK.axis["right"] = ax_ZWK.new_fixed_axis(loc="right", offset=(60, 0))
ax_SUM.axis["right"] = ax_SUM.new_fixed_axis(loc="right", offset=(60, 0))


ax_Z.axis["right"].toggle(all=True)
# ax_ZWK.axis["right"].toggle(all=True)
ax_SUM.axis["right"].toggle(all=True)


# 画图并获取返回值

.plot(time, Z, 'r-', label='Waterlevel')
# lns_ZWK, = ax_ZWK.plot(time, Z_WK, 'g-', label='Reservoir level')
lns_SUM, = ax_SUM.plot(time, SUM, color='orange', linestyle='-', label='Rainfall')

# 设置坐标范围
ax_Q.set_ylim(1.08*np.max(Q), np.min(Q))
ax_Z.set_ylim(np.min(Z), 1.08*np.max(Z))
# ax_ZWK.set_ylim(np.min(Z_WK), 1.08*np.max(Z_WK))
ax_SUM.set_ylim(1.38*np.max(SUM), np.min(SUM))

# 设置坐标名称
ax_Q.set_xlabel("Time (h)", )
ax_Q.set_ylabel("Runoff (m$^3$/s)", color='dodgerblue')
ax_Z.set_ylabel("Waterlevel (m)", color='r')
# ax_ZWK.set_ylabel("Reservoir level (m)", color='g')
ax_SUM.set_ylabel("Rainfall (mm)", color='orange')

# 画图例
lines = [lns_Q, lns_Z, lns_SUM]
ax_Q.legend(lines, [l.get_label() for l in lines])

# 设置标题
ax_Q.set_title('Hydrological or meteorological variables in Xixian', fontsize=30)
# 横坐标刻度
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1000))
"""
ax3 = fig.add_subplot(111)
ax3.plot(np.arange(0, 400, 1), corr_wy_sumq, label='Rainfall')
# ax3.plot(np.arange(0, 400, 1), corr_wy_zwkq, label='Reservoir level')
ax3.plot(np.arange(0, 400, 1), corr_wy_zq, label='Waterlevel')
ax3.set_xlabel('Time lag (h)')
ax3.set_ylabel('Correlation coefficient')

# ax3.plot(signal.argrelextrema(corr_wy_zq, np.greater, order=12)[0], corr_wy_zq[signal.argrelextrema(corr_wy_zq, np.greater, order=12)], '*', markersize=10, label='Maxima')  #极大值点
# ax3.plot(signal.argrelextrema(corr_wy_zq, np.less, order=12)[0], corr_wy_zq[signal.argrelextrema(corr_wy_zq, np.less, order=12)], 'o', markersize=10, label='Minima')  #极小值点
ax3.legend()

ax3.axhline(y=0.4, ls="--", c="r")  # 添加水平直线
ax3.plot(73, 0.4, '^', markersize=10)
ax3.plot(44, 0.4, 'o', markersize=10)
# ax3.plot(13, 0.4, '*', markersize=10)

ax3.text(75, 0.47, (73, 0.4))
ax3.text(44, 0.42, (44, 0.4))
# ax3.text(13, 0.42, (13, 0.4))

plt.subplots_adjust(wspace=0.3, hspace=0)
plt.title('The correlation coefficient between features and runoff', fontsize=30)

plt.grid()
plt.show()



