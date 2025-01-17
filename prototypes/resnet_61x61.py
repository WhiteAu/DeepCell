from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, merge, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
import cropping 
import h5py

reg = 1e-5
drop = 0.5
init = 'glorot_uniform'#'he_normal'
n_channels = 1
x_pixel = 61
y_pixel = 61

#Defines block creating function
def residual_block(block_function, n_filters, kernel, reps):
	def f(input):
		for i in range(reps):
			input = block_function(n_filters = n_filters, kernel=kernel)(input)
		return input

	return f

def res_block(block_function, reps):
	def f(input):
		for i in range(reps):
			input = block_function(input)
		return input

	return f


''' Define different resnet unit blocks '''
def residual_unit_3L(n_filters):
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act1)
		
		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		norm3 = BatchNormalization(axis = 1)(conv2)
		act3 = Activation('relu')(norm3)
		conv3 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act3)
		
		return merge([input, conv3], mode = "sum")

	return f


def residual_unit_1L(n_filters, kernel):
	def f(input):
		norm1 = BatchNormalization(axis = 1, mode = 2)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, kernel, kernel, init=init, border_mode = 'valid', W_regularizer = l2(reg))(act1)

		crop_size = (kernel - 1)/2
		#short1 = Convolution2D(n_filters, 3, 3, init=init, border_mode = 'valid', W_regularizer = l2(reg))(input)
		short1 = cropping.Cropping2D(cropping=((crop_size,crop_size),(crop_size,crop_size)) )(input)
		return merge([short1, conv1], mode="sum")

	return f


def residual_unit_1L2(n_filters, kernel, conv=False):
	def f(input):
		#convolutional path
		norm1 = BatchNormalization(axis = 1, mode = 2)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, kernel, kernel, init=init, border_mode = 'valid', W_regularizer = l2(reg))(act1)

		#shortcut:
		shortcut_input = input
		#if number of filter increases, set conv=True
		if conv:
			shortcut_input = Convolution2D(n_filters, 1, 1, init=init, border_mode = 'valid', W_regularizer = l2(reg))(input)
		crop_size = (kernel - 1)/2
		short1 = cropping.Cropping2D(cropping=((crop_size,crop_size),(crop_size,crop_size)) )(shortcut_input)
		return merge([short1, conv1], mode="sum")

	return f

def residual_unit_2L(n_filters):
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act1)
		
		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		return merge([input, conv2], mode = "sum")

	return f

def bottleneck_unit(n_filters):
	#follows the design from http://arxiv.org/pdf/1512.03385v1.pdf
	def f(input):
		norm1 = BatchNormalization(axis = 1)(input)
		act1 = Activation('relu')(norm1)
		conv1 = Convolution2D(n_filters, 1, 1, init=init, border_mode='same', W_regularizer = l2(reg))(act1)

		norm2 = BatchNormalization(axis = 1)(conv1)
		act2 = Activation('relu')(norm2)
		conv2 = Convolution2D(n_filters, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act2)

		norm3 = BatchNormalization(axis = 1)(conv2)
		act3 = Activation('relu')(norm3)
		conv3 = Convolution2D(n_filters*4, 3, 3, init=init, border_mode='same', W_regularizer = l2(reg))(act3)

		#need to convolve the input to change depth shape for merge to be valid
		short1 = Convolution2D(n_filters*4, 1, 1, init=init, border_mode='same', W_regularizer = l2(reg))(input)

		merge1 = merge([short1, conv3], mode = "sum")

		act4 = Activation('relu')(merge1)

		return act4

	return f

''' Define different resnet architectures '''
def resnet_61x61_1(n_channels, n_categories, n_unit1 = 1, n_unit2 = 1, n_unit3 = 1):
	input = Input(shape=(n_channels,61,61))

	conv1 = Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	#now the shape = (64, 59, 59)
	conv2 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	#now shape = (64, 56, 56)
	block1 = residual_block(residual_unit_2L, 64, n_unit1)(act2)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#now shape = (64, 28, 28)
	conv3 = Convolution2D(128, 1, 1, init = init, border_mode = 'same', W_regularizer = l2(reg))(pool1)
	#now shape = (128, 28, 28)
	block2 = residual_block(residual_unit_2L, 128, n_unit2)(conv3)
	pool2 = MaxPooling2D(pool_size=(2,2))(block2)
	#now shape = (128, 14, 14)
	conv4 = Convolution2D(256, 1, 1, init = init, border_mode = 'same', W_regularizer = l2(reg))(pool2)
	#now shape = (256, 14, 14)
	block3 = residual_block(residual_unit_2L, 256, n_unit3)(conv4)
	pool3 = MaxPooling2D(pool_size=(2,2))(block3)
	#now shape = (64, 7, 7)

	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=200, init = init, activation = "relu", W_regularizer = l2(reg))(flatten1)
	dense2 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(dense1)

	model = Model(input = input, output = dense2)

	return model

def resnet_61x61_2(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	#(1,61,61)
	conv1 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	#(64, 58, 58)
	block1 = residual_block(residual_unit_1L, 64, 3, 1)(conv1)
	#(64, 56, 56)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#(64, 28, 28)
	conv2 = Convolution2D(128, 1, 1, init = init, border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	#(128, 28, 28)
	block2 = residual_block(residual_unit_1L, 128, 3, 1)(conv2)
	#(128, 26, 26)
	block3 = residual_block(residual_unit_1L, 128, 3, 1)(block2)
	#(128, 24, 24)
	pool2 = MaxPooling2D(pool_size = (2,2))(block3)
	#(128, 12, 12)
	conv3 = Convolution2D(256, 1, 1, init = init, border_mode = 'valid', W_regularizer = l2(reg))(pool2)
	#(256, 12, 12)
	block4 = residual_block(residual_unit_1L, 256, 3, 1)(conv3)
	#(256, 10, 10)
	pool3 = MaxPooling2D(pool_size = (2,2))(block4)
	#(256, 5, 5)
	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model

def resnet_61x61_3(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	#(1,61,61)
	conv1 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	#(64, 58, 58)
	block1 = res_block(residual_unit_1L2(64, 3, conv=True), 1)(conv1)
	#(64, 56, 56)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#(64, 28, 28)
	block2 = res_block(residual_unit_1L2(128, 3, conv=True), 1)(pool1)
	#(128, 26, 26)
	block3 = res_block(residual_unit_1L2(128, 3), 1)(block2)
	#(128, 24, 24)
	pool2 = MaxPooling2D(pool_size = (2,2))(block3)
	#(128, 12, 12)
	block4 = res_block(residual_unit_1L2(256, 3, conv=True), 1)(pool2)
	#(256, 10, 10)
	pool3 = MaxPooling2D(pool_size = (2,2))(block4)
	#(256, 5, 5)
	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model

def resnet_61x61_4(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	#(1,61,61)
	conv1 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	#(64, 58, 58)
	block1 = res_block(residual_unit_1L2(64, 3, conv=True), 1)(conv1)
	#(64, 56, 56)
	block2 = res_block(residual_unit_1L2(64, 3), 1)(block1)
	#(64, 54, 54)
	block3 = res_block(residual_unit_1L2(64, 3), 1)(block2)
	#(64, 52, 52)
	block4 = res_block(residual_unit_1L2(64, 3), 1)(block3)
	#(64, 50, 50)
	block5 = res_block(residual_unit_1L2(64, 3), 1)(block4)
	#(64, 48, 48)
	pool1 = MaxPooling2D(pool_size = (2,2))(block4)
	#(64, 24, 24)
	block6 = res_block(residual_unit_1L2(128, 3, conv=True), 1)(pool1)
	#(128, 22, 22)
	block7 = res_block(residual_unit_1L2(128, 3), 1)(block6)
	#(128, 20, 20)
	pool2 = MaxPooling2D(pool_size = (2,2))(block7)
	#(128, 10, 10)
	block8 = res_block(residual_unit_1L2(256, 3, conv=True), 1)(pool2)
	#(256, 8, 8)
	pool3 = MaxPooling2D(pool_size = (2,2))(block8)
	#(256, 4, 4)
	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model

def resnet_61x61_5(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	#(1,61,61)
	conv1 = Convolution2D(16, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	#(16, 58, 58)
	block1 = res_block(residual_unit_1L2(16, 3), 7)(conv1)
	#(16, 44, 44)
	pool1 = MaxPooling2D(pool_size = (2,2))(block1)
	#(16, 22, 22)
	conv1 = Convolution2D(32, 1, 1, init = init, border_mode = 'valid', W_regularizer = l2(reg))(pool1)
	#(32, 22, 22)
	block2 = res_block(residual_unit_1L2(32, 3), 9)(conv1)
	#(32, 4, 4)
	flatten1 = Flatten()(block2)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model

def deepresnet_61x61(n_channels, n_categories):
	input = Input(shape=(n_channels,61,61))
	conv1 = Convolution2D(32, 1, 1, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	block1 = res_block(residual_unit_1L2(32, 3), 28)(conv1)
	#(32, 5, 5)
	flatten1 = Flatten()(block1)
	dense1 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(flatten1)
	model = Model(input = input, output = dense1)

	return model

def bottle_net_61x61(n_channels, n_categories, n_unit1 = 1, n_unit2 = 1, n_unit3 = 1):
	#inspired by http://arxiv.org/pdf/1512.03385v1.pdf
	input = Input(shape=(n_channels,61,61))

	conv1 = Convolution2D(64, 3, 3, init = init, border_mode = 'valid', W_regularizer = l2(reg))(input)
	norm1 = BatchNormalization(axis = 1)(conv1)
	act1 = Activation('relu')(norm1)
	#now the shape = (64, 59, 59)
	conv2 = Convolution2D(64, 4, 4, init = init, border_mode = 'valid', W_regularizer = l2(reg))(act1)
	norm2 = BatchNormalization(axis = 1)(conv2)
	act2 = Activation('relu')(norm2)
	#now shape = (64, 56, 56)
	block1 = res_block(bottleneck_unit(64), n_unit1)(act2)
	pool1 = MaxPooling2D(pool_size=(2,2))(block1)
	#now shape = (64, 28, 28)
	block2 = res_block(bottleneck_unit(64), n_unit2)(pool1)  #(conv3)
	pool2 = MaxPooling2D(pool_size=(2,2))(block2)
	#now shape = (128, 14, 14)
	block3 = res_block(bottleneck_unit(64), n_unit3)(pool2)  #(conv4)
	pool3 = MaxPooling2D(pool_size=(2,2))(block3)
	#now shape = (64, 7, 7)

	flatten1 = Flatten()(pool3)
	dense1 = Dense(output_dim=200, init = init, activation = "relu", W_regularizer = l2(reg))(flatten1)
	dense2 = Dense(output_dim=n_categories, init = init, activation = "softmax", W_regularizer = l2(reg))(dense1)

	model = Model(input = input, output = dense2)

	return model





