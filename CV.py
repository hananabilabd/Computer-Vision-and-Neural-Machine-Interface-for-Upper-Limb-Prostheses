# coding: utf-8

# In[86]:


# PROBLEMS:
#  1-EMG: fail to predict the right class
# 2-Error in the CV
# https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
# https://stackoverflow.com/questions/49287934/dask-dataframe-prediction-of-keras-model/49290185?noredirect=1#comment85587469_49290185

# create new h5 in backend Theano


# !pip install -q keras
import keras
import h5py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from scipy import misc, ndimage
from skimage import io
import random
from itertools import chain
from sklearn.preprocessing import LabelBinarizer
import io
from sklearn.externals import joblib
import pickle
# keras pakages
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
    AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, LeakyReLU
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
# get_ipython().magic(u'matplotlib inline')
from keras.models import model_from_json
import keras.backend as K

K.set_image_data_format( 'channels_last' )
K.set_learning_phase( 1 )

import threading
import os
import numpy as np
import cv2
import time
from collections import Counter
import queue  ##If python 3
# import Queue as queue ##If python 2
import scipy.io as sio
from scipy.signal import butter, lfilter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from scipy import stats

class CV():
    def __init__(self, queue_size=8):
        print("Hi People")
        self.q = queue.Queue()

    def rgb2gray(self,rgb_image):
        return np.dot( rgb_image, [0.299, 0.587, 0.114] )


    def real_preprocess(self,img):
        # gray level
        img_gray = self.rgb2gray( img )

        # resize the image 48x36:
        img_resize = misc.imresize( img_gray, (48, 36) )

        # Normalization:
        img_norm = (img_resize - img_resize.mean()) / img_resize.std()

        return img_norm


    def Nazarpour_model(self,input_shape, num_of_layers=2):
        x_input = Input( input_shape )

        x = Conv2D( 5, (5, 5), strides=(1, 1), padding='valid' )( x_input )
        x = BatchNormalization( axis=3 )( x )
        x = Activation( 'relu' )( x )
        x = Dropout( 0.2 )( x )

        if num_of_layers == 2:
            x = Conv2D( 25, (5, 5), strides=(1, 1), padding='valid' )( x )
            x = BatchNormalization( axis=3 )( x )
            x = Activation( 'relu' )( x )

        x = MaxPooling2D( (2, 2), strides=(2, 2) )( x )
        x = Dropout( 0.2 )( x )

        x = Flatten()( x )

        x = Dense( 4, activation='softmax', kernel_initializer=glorot_uniform( seed=0 ) )( x )

        model = Model( inputs=x_input, outputs=x )

        return model


    def grasp_type(self,path_of_test_real, model_name):
        """
        path_of_test_real : the path of the uploaded image in case of offline.
        model_name: the name of the trained model, 'tmp.h5'

        """

        n_row = 48
        n_col = 36
        nc = 1
        model = self.Nazarpour_model( (n_row, n_col, nc), num_of_layers=2 )
        model.compile( 'adam', loss='categorical_crossentropy', metrics=['accuracy'] )
        model.load_weights( model_name )

        i = misc.imread( path_of_test_real )
        img_after_preprocess = self.real_preprocess( i )
        x = np.expand_dims( img_after_preprocess, axis=0 )
        x = x.reshape( (1, n_row, n_col, nc) )
        out = model.predict( x )
        grasp = np.argmax( out ) + 1

        return grasp

    def Main_algorithm(self,path1,path2=None):
        #    event.wait()
        stage = 0  # I changed it to random number not zero just for test. retrun it back!!
        corrections = 0
        all_grasps = [1, 2, 3, 4]
        Choose_grasp = list( all_grasps )

        self.path_of_real_test = path1  # put the path of the tested picture
        if path2:
            self.model_name = path2
        else :
            self.model_name='tools/class 1/50_r110.png'

        while not(q.empty()):  # not(q.empty())):
            EMG_class_recieved = q.get()
            if (EMG_class_recieved == 0 or stage == 0):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, stage ))
                System_power( 1 )  # Start system

            elif (EMG_class_recieved == 1):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, stage ))
                Confirmation()

            elif (EMG_class_recieved == 2):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, stage ))
                Cancellation()

            elif (EMG_class_recieved == 3):
                print("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, stage ))
                System_power( 0 )  # Turn system off

    def System_power(self,Turn_on):
        # Reset values:
        stage = 0
        corrections = 0
        Choose_grasp = list( all_grasps )

        if not Turn_on:
            # Turn off
            print ("Turning off ... \n\n\n")
        else:
            # Restart
            print ("Restarting ... \n")
            grasp = grasp_type( self.path_of_real_test, self.model_name )
            print ('Grasp type no.{0} \n'.format( grasp ))
            stage = 1

    def Confirmation(self):
        print("    Confirmed! \n")
        if stage < 3:
            stage += 1
            corrections = 0
            Choose_grasp = list( all_grasps )
            print ('Grasping \n')
            # Do the action
        elif stage == 2:
            print ('Releasing ... \n')
            System_power( 0 )

    def Cancellation(self):
        if stage > 0:
            if (stage == 2 and corrections > 3):
                print("Exceeded maximum iteration: \n Choosing from remaining grasps")
                Choose_grasp.remove( grasp )
                # Choose random class
            else:
                print("    Cancelled! \n")
                stage -= 1
                corrections += 1
                # Redo previous action
        else:
            print ('No previous stage, restarting ... \n')
            System_power( 1 )


#q = queue.Queue()



"""
Stages meanings:
0: System off
1: Taking photos, deciding grasp type, preshaping.
2: Grasping
3: Releasing
"""

cv =CV()



# t1 = threading.Thread(target = EMG_Listener, name ='thread1')
# t2 = threading.Thread(target = Main_algorithm, name ='thread2')

# t1.daemon = True
# t2.daemon = True

# t1.start()
# t2.start()

# t1.join()
grasp = cv.grasp_type( 'tools/class 1/50_r110.png', 'tools/GP_Weights.h5' )
print ('Grasp type no.{0} \n'.format( grasp ))

