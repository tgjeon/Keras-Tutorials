from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.regularizers import *


def get_model():
	Input_1 = Input(shape=(28, 28, 1))
	Conv2D_6 = Conv2D(activation= 'relu' ,nb_col= 3,nb_filter= 32,nb_row= 3,border_mode= 'same' )(Input_1)
	MaxPooling2D_4 = MaxPooling2D(strides= (2,2),border_mode= 'same' ,pool_size= (2,2))(Conv2D_6)
	Dropout_1 = Dropout(p= 0.3)(MaxPooling2D_4)
	Conv2D_8 = Conv2D(nb_col= 3,nb_filter= 64,nb_row= 3,border_mode= 'same' ,init= 'glorot_normal' ,activation= 'relu' )(Dropout_1)
	MaxPooling2D_6 = MaxPooling2D(strides= (2,2),border_mode= 'same' ,pool_size= (2,2))(Conv2D_8)
	Dropout_2 = Dropout(p= 0.3)(MaxPooling2D_6)
	Conv2D_9 = Conv2D(nb_col= 3,nb_filter= 128,nb_row= 3,border_mode= 'same' ,init= 'glorot_normal' ,activation= 'relu' )(Dropout_2)
	MaxPooling2D_7 = MaxPooling2D(border_mode= 'same' ,pool_size= (2,2))(Conv2D_9)
	Flatten_2 = Flatten()(MaxPooling2D_7)
	Dropout_3 = Dropout(p= 0.3)(Flatten_2)
	Dense_4 = Dense(activation= 'relu' ,init= 'glorot_normal' ,output_dim= 625)(Dropout_3)
	Dropout_4 = Dropout(p= 0.5)(Dense_4)
	Dense_5 = Dense(activation= 'softmax' ,output_dim= 10)(Dropout_4)

	return Model([Input_1],[Dense_5])


from keras.optimizers import *

def get_optimizer():
	return Adadelta()

def get_loss_function():
	return 'categorical_crossentropy'

def get_batch_size():
	return 64

def get_num_epoch():
	return 10

def get_data_config():
	 return {"mapping": {"Image": {"port": "InputPort0", "type": "Image"}, "Digit Label": {"port": "OutputPort0", "type": "Categorical"}}, "samples": {"split": 1, "validation": 14000, "test": 14000, "training": 42000}, "dataset": {"samples": 70000, "type": "public", "name": "mnist"}, "datasetLoadOption": "batch", "numPorts": 1}