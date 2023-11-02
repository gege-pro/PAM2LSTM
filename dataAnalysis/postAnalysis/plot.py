import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import numpy as np


plt.rc('font', family='Times New Roman', size=24)
plt.figure(figsize=(10, 5))
flights = pd.read_csv("att.csv")
flights = flights.pivot("Network", "Time step (h)", "weight1")
sns.heatmap(flights, linewidths=0.2, cmap="YlGnBu", annot=False, annot_kws={"fontsize": 20})
plt.show()

"""
# 累计误差增长率
t_recur = [0, 0.25, 2.5, 5.75]
t_direct = [0, 0.25, 1.75, 5.5]
r_recur = [0, 84.62, 55.22, 525.33]
r_direct = [0, -14.51, 36.57, 86.97]
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_recur, '--o', color='b', label='Time shift LSTM')
ax.plot(t_direct, '-v', color='r', label='Time shift our model')
ticks = ax.set_xticks([0, 1, 2, 3])  # 设置刻度
labels = ax.set_xticklabels(['T+1', 'T+2', 'T+6', 'T+12'], rotation=0, fontsize=20)  # 设置刻度标签
ax.set_xlabel('Time step (hour)')
ax.set_ylabel('The difference of Time shift (hour)')
ax.grid()
# 两根Y轴
ax2 = ax.twinx()
ax2.plot(r_recur, '-.x', color='g', label='Runoff shift LSTM')
ax2.plot(r_direct, '-s', color='y', label='Runoff shift our model')
ax2.set_ylabel('The rate of change in runoff shift (%)')
fig.legend(['Time shift LSTM', 'Time shift our model', 'Runoff shift LSTM', 'Runoff shift our model'], fontsize=20, loc='upper left', bbox_to_anchor=(0.13,0.9))
plt.show()
"""

"""
#
preAndReal = pd.read_csv('LSTM_P&S(1).csv')
preAndRealwy = pd.read_csv('LSTMwy_P&S(1).csv')
fig, axes = plt.subplots(1, 2)
ax1 = axes[0]
ax1.scatter(preAndReal['real'], preAndReal['pred'], marker='o', color='darkorange')
ax1.plot(preAndReal['real'], preAndReal['real'], color='black', linewidth=2, linestyle='-.')
x = np.linspace(2623, 3227)
x1 = preAndReal['real']
y_err = x1.std() * np.sqrt(1/len(x1) + (x1 - x1.mean())**2 / np.sum((x1 - x1.mean())**2))
print(np.mean(y_err))
y1 = x-(3227-2623)*np.mean(y_err)*0.1
y2 = x+(3227-2623)*np.mean(y_err)*0.1
ax1.fill_between(x, y1, y2, color='blue', alpha=0.15)
ax1.text(31.5, 36, s="MAE=0.0180", color='b')
ax1.text(31.5, 35, s="RMSE=0.0504", color='b')
ax1.text(31.5, 34, s="NSE=0.9965", color='b')
ax1.set_xlabel('Observation/ (m$^3$/s)')
ax1.set_ylabel('Forecast/ (m$^3$/s)')
ax1.set_title('XiXian')
ax1.grid()

ax2 = axes[1]
ax2.scatter(preAndRealwy['real'], preAndRealwy['pred'], marker='o', color='darkorange')
ax2.plot(preAndRealwy['real'], preAndRealwy['real'], color='black', linewidth=2, linestyle='-.')
x = np.linspace(2300, 2630)
y1 = x-2630*0.01
y2 = x+2630*0.01
plt.fill_between(x, y1, y2, color='blue', alpha=0.15)
ax2.text(54.5, 60, s="MAE=0.0385", color='b')
ax2.text(54.5, 59, s="RMSE=0.0963", color='b')
ax2.text(54.5, 58, s="NSE=0.9785", color='b')
ax2.set_xlabel('Observation/ (m$^3$/s)')
ax2.set_ylabel('Forecast/ (m$^3$/s)')
ax2.set_title('WuYuan')
ax2.grid()
plt.show()
"""

