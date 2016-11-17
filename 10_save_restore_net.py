from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# Hyper parameters
batch_size = 128
nb_epoch = 10

# Parameters for MNIST dataset
nb_classes = 10

# Parameters for MLP
prob_drop_input = 0.2               # drop probability for dropout @ input layer
prob_drop_hidden = 0.5              # drop probability for dropout @ fc layer


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_Train = np_utils.to_categorical(y_train, nb_classes)
Y_Test = np_utils.to_categorical(y_test, nb_classes)

# Multilayer Perceptron model
model = Sequential()
model.add(Dense(output_dim=625, input_dim=784, init='normal', activation='sigmoid', name='dense1'))
model.add(Dropout(prob_drop_input, name='dropout1'))
model.add(Dense(output_dim=625, input_dim=625, init='normal', activation='sigmoid', name='dense2'))
model.add(Dropout(prob_drop_hidden, name='dropout2'))
model.add(Dense(output_dim=10, input_dim=625, init='normal', activation='softmax', name='dense3'))
model.compile(optimizer=RMSprop(lr=0.001, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
save_model(model, './logs/model_mlp')
checkpoint = ModelCheckpoint(filepath='./logs/weights.epoch.{epoch:02d}-val_loss.{val_loss:.2f}.hdf5', verbose=0)
history = model.fit(X_train, Y_Train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                    callbacks=[checkpoint], validation_data=(X_test, Y_Test))

# Evaluate
evaluation = model.evaluate(X_test, Y_Test, verbose=1)
print('\nSummary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

# Restore trained model
loaded_model = load_model('./logs/model_mlp')
loaded_model.load_weights('./logs/weights.epoch.09-val_loss.0.08.hdf5')
loaded_model.summary()

# Evaluate with loaded model
evaluation = loaded_model.evaluate(X_test, Y_Test, verbose=1)
print('\nSummary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))