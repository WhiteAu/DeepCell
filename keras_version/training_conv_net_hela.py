'''Train a simple deep CNN on a HeLa dataset.
GPU run command:
	THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python training_template.py

'''

from __future__ import print_function
from keras.optimizers import SGD, RMSprop

from cnn_functions import rate_scheduler, train_model_sample
from model_zoo import bn_feature_net_61x61 

import os
import datetime
import numpy as np
import keras

batch_size = 256
n_epoch = 25

dataset = "HeLa_all_61x61"
expt = "bn_feature_net_61x61"

usr_home = os.path.expanduser('~')
root_dir = os.path.join(usr_home, "projects/deepcell")
direc_save = os.path.join(root_dir, "trained_networks/HeLa")
direc_data = os.path.join(root_dir, "training_data_npz/HeLa")

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
lr_sched = rate_scheduler(lr = 0.01, decay = 0.95)
class_weight = {0:1, 1:1, 2:1}

for iterate in xrange(5):

	model = bn_feature_net_61x61(n_channels = 2, n_features = 3, reg = 1e-5)

	train_model_sample(model = model, 
                           dataset = dataset, 
                           optimizer = optimizer, 
                           expt = expt, 
                           it = iterate, 
                           batch_size = batch_size, 
                           n_epoch = n_epoch,
                           direc_save = direc_save, 
                           direc_data = direc_data, 
                           lr_sched = lr_sched, 
                           class_weight = class_weight,
                           rotate = True, 
                           flip = True, 
                           shear = False)

	del model
	from keras.backend.common import _UID_PREFIXES
	for key in _UID_PREFIXES.keys():
		_UID_PREFIXES[key] = 0

