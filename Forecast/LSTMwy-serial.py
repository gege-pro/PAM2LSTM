import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras_preprocessing import sequence
from scipy import signal
from sklearn.metrics import mean_squared_error, mean_absolute_error


plt.rc('font', family='Times New Roman', size=24)


def create_dataset(data, look_back1, look_back2, look_back3, look_back4, pre_len):
    """
    create dataset
    :param data: raw data
    :param look_back1: feature1 length
    :param look_back2: feature2 length
    :param look_back3: feature3 length
    :param pre_len: prediction length
    :return: dataset
    """
    X, Y = [], []
    print("data:", data.shape)
    maxlen = max(look_back1, look_back2, look_back3, look_back4)
    for i in range(len(data) - maxlen - pre_len - 1):
        temp1 = data[i + maxlen - look_back1:i + maxlen, 0].tolist()  # R
        temp2 = data[i + maxlen - look_back2:i + maxlen, 1].tolist()  # Q
        temp3 = data[i + maxlen - look_back3:i + maxlen, 2].tolist()  # Z_WK
        temp4 = data[i + maxlen - look_back4:i + maxlen, 3].tolist()  # Z
        temp = [temp1, temp2, temp3, temp4]
        # temp = sequence.pad_sequences(temp, maxlen=max(look_back1, look_back2, look_back3), value=0)
        X.append(temp)
        Y.append(data[i + maxlen:i + maxlen + pre_len, 3])
    X = np.array(X)
    Y = np.array(Y)
    print("X.shape", X.shape)
    X = np.reshape(X, newshape=(X.shape[0], maxlen, 4))
    return X, Y


def inverse_transform_col(scaler, y, n_col):
    """
    inverse transform
    :param scaler:
    :param y: feature
    :param n_col: number
    :return:
    """
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y


def plot_all(data, len_pre):
    """
    plot
    :param data:
    :param len_pre:
    :return:
    """
    temp11 = np.array(data)
    all_pre = temp11.reshape((-1, len_pre))
    # print("all_pre:", all_pre.shape, all_pre)
    row11 = all_pre.shape[0]
    col11 = all_pre.shape[1]
    pre_filp = np.fliplr(all_pre)
    Prediction_Z = []
    for i in range(-row11+1, col11):
        diag = np.diagonal(pre_filp, offset=i)
        diag_mu = diag.mean()
        Prediction_Z.append(diag_mu)
    Prediction_Z.reverse()
    return Prediction_Z


#
# data = pd.read_csv('../dataSrc/xx_2010_2018.csv', encoding='gbk')
data = pd.read_csv('../dataSrc/wy_2017_2021.csv', encoding='gbk')
# data = data[['SUM', 'Q', 'Z']]
data = data[['SUM', 'Q', 'Z_WK', 'Z']]

# transform
scaler = MinMaxScaler()
data = scaler.fit_transform(data)  # numpy.array

# set parameter
len_R = 72
len_Q = 72
len_WK = 72
len_Z = 72
MaxLen = max(len_R, len_Q, len_WK, len_Z)
len_pre = 6
Batch_size = 64
EPOCH = 100

#
# X_set, Z_set = create_dataset(data, len_R, len_Z, len_pre)
X_set, Z_set = create_dataset(data, len_R, len_Q, len_WK, len_Z, len_pre)

# split dataset
ratio = 0.8
length = int(len(X_set) * ratio)

X_trainset = X_set[:length]
X_testset = X_set[length:]

Z_trainset = Z_set[:length]
Z_testset = Z_set[length:]

input_layer = tf.keras.layers.Input(shape=(MaxLen, 4))
# x = tf.keras.layers.Masking(mask_value=0, input_shape=(MaxLen, 4))
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=180, activation='tanh', return_sequences=True))(input_layer)
# x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=80, activation='tanh', return_sequences=True))(x)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, activation='tanh', return_sequences=False))(x)
x = tf.keras.layers.Flatten()(x)

#
output_layer = tf.keras.layers.Dense(len_pre, activation="relu")(x)

GRU_layer = tf.keras.Model(inputs=input_layer, outputs=output_layer)
GRU_layer.summary()

adam = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)
GRU_layer.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)


# train
history = GRU_layer.fit(X_trainset, Z_trainset, batch_size=Batch_size, epochs=EPOCH, verbose=1,
                        validation_split=0.3, shuffle=False, initial_epoch=0, callbacks=[early_stopping])
# save model
GRU_layer.save('../h5/BiLSTMwy' + '-serial' + str(len_pre) + '.h5')
history_dict = history.history
loss_value = history_dict["loss"]
val_loss_value = history_dict["val_loss"]
np.save('../h5/loss' + '-' + str(len_pre) + '.npy', loss_value)
np.save('../h5/val_loss' + '-' + str(len_pre) + '.npy', val_loss_value)

loss_value = np.load('../h5/loss' + '-' + str(len_pre) + '.npy')
val_loss_value = np.load('../h5/val_loss' + '-' + str(len_pre) + '.npy')
plt.plot(loss_value[:], label='loss')
plt.plot(val_loss_value[:], label='val_loss')
plt.legend()
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# predict
GRU_layer.load_weights('../h5/BiLSTMwy' + '-serial' + str(len_pre) + '.h5')
pre_Z = GRU_layer.predict(X_testset)
#
Prediction_Z = plot_all(pre_Z, len_pre)
Real_Z = plot_all(Z_testset, len_pre)
#
Prediction_Z = inverse_transform_col(scaler, Prediction_Z, 3)
Real_Z = inverse_transform_col(scaler, Real_Z, 3)
#
preAndReal = pd.DataFrame()
preAndReal['pred_s'] = pd.DataFrame(Prediction_Z)
preAndReal['real'] = pd.DataFrame(Real_Z)
preAndReal.to_csv('../h5/BiLSTM_P&S1.csv')

MAE = mean_absolute_error(preAndReal['real'], preAndReal['pred_s'])
RMSE = np.sqrt(mean_squared_error(preAndReal['real'], preAndReal['pred_s']))
SS_R = sum((preAndReal['real']-preAndReal['pred_s'])**2)
SS_T = sum((preAndReal['real']-np.mean(preAndReal['real']))**2)
NSE = 1-(float(SS_R))/SS_T
print("Parallel:", MAE, RMSE, NSE)




