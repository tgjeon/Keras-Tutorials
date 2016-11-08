from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

# Generate dataset
trX = np.linspace(-1, 1, 101)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33 # create a y value which is approximately linear but with some random noise

# Linear regression model
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1, init='normal', activation='linear'))
model.compile(optimizer=SGD(lr=0.01), loss='mean_squared_error', metrics=['accuracy'])

# Train
model.fit(trX, trY, nb_epoch=100, verbose=1)