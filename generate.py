
import sys  # For command line args
import pandas as pd  # For saving data
import numpy as np  # For preprocessing data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

import joblib  # for loading preprocessing info

# Load preprocessing info
dataX = joblib.load("data/dataX")
d_mean = joblib.load("data/data_mean")
d_std = joblib.load("data/data_std")

# Ensure correct number of arguments
if len(sys.argv) != 3:
    print("Usage: python generate.py <weights_file> <num_moves>")
    sys.exit(1)

# Weights file and number of moves
weights_file = sys.argv[1]
num_moves = int(sys.argv[2])

# Define the model
model = Sequential()
model.add(LSTM(512, input_shape=(5, 26), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dropout(0.2))
model.add(Dense(26, activation='linear'))

# Compile the model
model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam')

# Load weights
model.load_weights(weights_file)
print("Model loaded")

# Randomly select a starting position
start = np.random.randint(0, len(dataX) - 1)
pattern = dataX[start]

# Generate moves
moves = []
for i in range(num_moves):
    print(f"Generating step {i+1}")
    x = np.reshape(pattern, (1, len(pattern), 26))
    new_move = model.predict(x)
    moves.append(new_move[0])
    pattern = np.append(pattern, new_move, axis=0)
    pattern = pattern[1:]  # Keep the most recent steps

# Convert back the normalized data
moves = np.array(moves)
moves = moves.reshape((-1, 26))
moves = moves * d_std + d_mean
newMoves = pd.DataFrame(moves)

# Save new moves
newMoves.to_csv("new_moves.csv", index=False, header=False)