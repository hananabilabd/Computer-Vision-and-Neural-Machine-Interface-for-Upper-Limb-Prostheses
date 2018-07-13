# PROBLEMS:
#  1-EMG: fail to predict the right class
# 2-Error in the CV
# https://github.com/keras-team/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
# https://stackoverflow.com/questions/49287934/dask-dataframe-prediction-of-keras-model/49290185?noredirect=1#comment85587469_49290185
# !pip install -q keras
#import keras
import h5py
import numpy as np
import random
import time
from scipy import misc #, ndimage
import queue  ##If python 3
# import Queue as queue ##If python 2
import threading
#import time
#import cv2
#import matplotlib.pyplot as plt
#import pandas as pd
#from skimage import io
#from itertools import chain
#from sklearn.preprocessing import LabelBinarizer
#import io
#from sklearn.externals import joblib
#import pickle
# keras pakages
#from keras import layers
#from keras.layers import   LeakyReLU , AveragePooling2D ,GlobalMaxPooling2D  ,ZeroPadding2D
###############
from keras.layers import Input, Add, Dense, Activation,Dropout , BatchNormalization, Flatten, Conv2D,MaxPooling2D
from keras.models import Model #, load_model
from keras.initializers import glorot_uniform
from keras import backend as K
import tensorflow as tf
############
#from keras.preprocessing import image
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
#from keras.applications.imagenet_utils import preprocess_input
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot
#from keras.utils import plot_model
#import scipy.misc
#from matplotlib.pyplot import imshow
# get_ipython().magic(u'matplotlib inline')
#from keras.models import model_from_json
#import keras.backend as K
#K.set_image_data_format( 'channels_last' )
#K.set_learning_phase( 1 )

#import os
#from collections import Counter
#import scipy.io as sio
#from scipy.signal import butter, lfilter, filtfilt
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from scipy import stats
#global graph
#graph = tf.get_default_graph()
class CV():
    def __init__(self, queue_size=8):
        self.q = queue.Queue()
        self.stage = 0
        self.corrections = 0
        self.all_grasps = [1, 2, 3, 4]
        self.Choose_grasp = list( self.all_grasps )
        self.grasp1=None
        self.graph = tf.get_default_graph()

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


    def grasp_type(self,path, Mname):
        """
        path_of_test_real : the path of the uploaded image in case of offline.
        model_name: the name of the trained model, 'tmp.h5'

        """
        #f = h5py.File( Mname, 'r' )
        #f.close()
        n_row = 48
        n_col = 36
        nc = 1

        model = self.Nazarpour_model( (n_row, n_col, nc), num_of_layers=2 )
        model.compile( 'adam', loss='categorical_crossentropy', metrics=['accuracy'] )
        model.load_weights(self.path1)
        i = misc.imread(self.model_name )
        img_after_preprocess = self.real_preprocess( i )
        x = np.expand_dims( img_after_preprocess, axis=0 )
        x = x.reshape( (1, n_row, n_col, nc) )
        out = model.predict( x )
        grasp = np.argmax( out ) + 1

        if grasp == 1 :
            print( ("Grasp_Type : Pinch \n ") )
        if grasp == 2 :
            print( ("Grasp_Type  : Palmar Wrist Neutral \n ") )
        if grasp == 3 :
            print( ("Grasp_Type  : Tripod \n ") )
        if grasp == 4 :
            print( ("Grasp_Type : Palmar Wrist Pronated \n ") )
        return grasp

    def kk (self):
        #if K.backend() == 'tensorflow':
        K.clear_session()
        tf.reset_default_graph()

    def new_session(self):
        if K.backend() == 'tensorflow':  # pragma: no cover
            import tensorflow as tf
            K.clear_session()
            config = tf.ConfigProto( allow_soft_placement=True )
            config.gpu_options.allow_growth = True
            session = tf.Session( config=config )
            K.set_session( session )

    def tt(self):
        threading.Thread( target= K.clear_session() ).start()
    def finish(self):
        import keras.backend.tensorflow_backend
        if keras.backend.tensorflow_backend._SESSION:
            import tensorflow as tf
            tf.reset_default_graph()
            keras.backend.tensorflow_backend._SESSION.close()
            keras.backend.tensorflow_backend._SESSION = None
        #from keras import backend as K
        #K.clear_session()




    def Main_algorithm(self,path1,path2=None):
        #f = h5py.File( '../tools/GP_Weights.h5', 'r' )
        #f.close()

        #self.path_of_real_test = path1  # put the path of the tested picture
        self.path1 = path1
        if path2:
            self.model_name = path2
        else:
            self.model_name = 'tools/class 1/50_r110.png'




        #    path_of_real_test='/home/ghadir/Downloads/__/class 1/50_r110.png' #put the path of the tested picture
        #    CV_model_name='GP_Weights.h5'
        # """

        while not (self.q.empty()):
            EMG_class_recieved = self.q.get()
            if (EMG_class_recieved == 1 or self.stage == 0):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.System_power( 1 )  # Start system

            elif (EMG_class_recieved == 2):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.Confirmation()

            elif (EMG_class_recieved == 3):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.Cancellation()

            elif (EMG_class_recieved == 0):
                print(("EMG_class {0}, Stage {1} : \n".format( EMG_class_recieved, self.stage )))
                self.System_power( 0 )  # Turn system off

    def System_power(self,Turn_on):



        # Reset values:
        self.stage = 0
        #    corrections= 0
        self.Choose_grasp = list( self.all_grasps )

        if not Turn_on:
            self.corrections = 0
            # Turn off
            print ("Turning off ... back to rest state. \n\n\n")
        else:
            # Start/restart
            with self.graph.as_default():
                self.grasp1 = self.grasp_type( self.path1, self.model_name )
            #self.grasp1 = self.grasp_type( 'ww.h5', '../tools/class 1/50_r110.png' )

            print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
            self.stage = 1

    def Confirmation(self):


        print("    Confirmed! \n")
        if self.stage < 2:
            self.stage += 1
            self.corrections = 0
            self.Choose_grasp = list( self.all_grasps )
            print(("Grasping ... grasp type{} \n\n").format( self.grasp1 ))
            # Do the action
        else:
            print ('Releasing ... \n')
            self.System_power( 0 )


    def Cancellation(self):



        if self.stage > 0:
            print("    Cancelled! \n")
            self.stage -= 1
            #        corrections +=1
            if (self.stage == 0 and self.corrections > 3):
                print("Exceeded maximum iteration: \n Choosing from remaining grasps")
                if self.Choose_grasp:
                    if self.grasp1 in self.Choose_grasp:
                        self.Choose_grasp.remove( self.grasp1 )
                if not self.Choose_grasp: #To check if list is empty after removing an element.
                    self.Choose_grasp = list( self.all_grasps )
                    self.corrections = 0
                self.grasp1 = random.SystemRandom().choice( self.Choose_grasp )
                print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
                self.stage = 1
            else:
                # Redo previous action:
                if self.stage == 0:
                    self.System_power( 1 )
                    self.corrections += 1
                    print ("Restarting ... \n")
                elif self.stage == 1:
                    print(('Preshaping grasp type {}\n\n').format( self.grasp1 ))
                elif self.stage == 2:
                    print(("Grasping ... grasp type{} \n\n").format( self.grasp1 ))
            print(("Correction no. {}").format( self.corrections + 1 ))


        else:
            print ('No previous stage, restarting ... \n')
            self.System_power( 1 )






            #q = queue.Queue()



"""
Stages meanings:
0: System off
1: Taking photos, deciding grasp type, preshaping.
2: Grasping
3: Releasing
"""





# t1 = threading.Thread(target = EMG_Listener, name ='thread1')
# t2 = threading.Thread(target = Main_algorithm, name ='thread2')

# t1.daemon = True
# t2.daemon = True

# t1.start()
# t2.start()

# t1.join()

#cv =CV()
#grasp = cv.grasp_type( 'tools/class 1/50_r110.png', 'tools/GP_Weights.h5' )
#print ('Grasp type no.{0} \n'.format( grasp ))
"""
cv.q.put(2)
cv.q.put(1)
while(1):
    #cv.kk()
    time.sleep(2)
    cv.q.put( 2 )
    cv.q.put( 1 )

    print (("Hiii"))
    #threading.Thread( target= cv.Main_algorithm(path1='tools/GP_Weights.h5')).start()
    cv.q.put(1)
    cv.q.put(1)
    #with graph.as_default():
    cv.Main_algorithm(path1='tools/GP_Weights.h5')

"""