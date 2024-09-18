import pandas as pd  # For loading data
import numpy as np  # For preprocessing data
import tensorflow as tf  # For DL
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import Dense, Dropout, LSTM  # Layers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard  # For saving weights and other callbacks
import joblib  # For dumping preprocessing parameters
import os

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# List of the named columns for the position data
poseList = [
    "nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow",
    "leftWrist", "rightWrist", "leftHip", "rightHip", "leftKnee",
    "rightKnee", "leftAnkle", "rightAnkle",
]

# Each data point has x and y coordinates, so we added extra columns for them
cols = [f"{pose}-x" for pose in poseList] + [f"{pose}-y" for pose in poseList]

# Reading data from csv files
dfs = [pd.read_csv(f"data/dance_download{i}.csv", header=None) for i in range(1, 6)]

# Merging all the data and assigning columns
df = pd.concat(dfs, ignore_index=True)
df.columns = cols

data = df.copy()

# Preprocessing data
data = np.array(data)
d_mean = data.mean(axis=0)
d_std = data.std(axis=0)

# Saving the mean and std of data
joblib.dump(d_mean, "data/data_mean")
joblib.dump(d_std, "data/data_std")

data = (data - d_mean) / d_std

# Maximum timesteps to feed into the LSTM
seq_length = 5
dataX, dataY = [], []

# Generating sequences of the data
for i in range(0, len(data) - seq_length, 1):
    seq_in = data[i:i + seq_length]
    seq_out = data[i + seq_length]
    dataX.append(seq_in)
    dataY.append(seq_out)
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

joblib.dump(dataX, "data/dataX")

# Reshaping and batching data
X = np.reshape(dataX, (n_patterns, seq_length, len(cols)))
y = np.array(dataY)

# Creating model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='linear'))
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='rmsprop')

model.summary()  # Prints the model summary

# File path for saving weights
filepath = "weights/weights-improvement-{epoch:02d}-{loss:.4f}.keras"

# Callback for saving weights
checkpoint = ModelCheckpoint(
    filepath, monitor='loss', verbose=1, save_best_only=True, mode='min'
)

# Callback for creating logs
tensorboard_callback = TensorBoard(log_dir="logs")

callbacks_list = [checkpoint, tensorboard_callback]

# Fitting / training the model
model.fit(X, y, epochs=1000, batch_size=128, callbacks=callbacks_list)
