from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Generate dataset
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

# Linear regression model
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1, init='uniform', activation='linear'))
model.compile(optimizer='sgd', loss='mse')

# Train
model.fit(trX, trY, nb_epoch=100, verbose=1)