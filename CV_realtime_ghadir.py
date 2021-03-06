#==============================================================================
# REAL LIFE CLASSIFICATION
#==============================================================================
import tensorflow as tf
from keras.applications import imagenet_utils ,ResNet50
#from keras.applications import VGG16
#from keras.applications import ResNet50
import cv2, threading
import numpy as np
import time
import random
import queue  ##If python 3
# import Queue as queue ##If python 2
l1 =[ "wooden_spoon" , "fountain_pen", "revolver" ,"kite" , "necklace" , "ballpoint"]
l2 =["beer_glass"  , "hourglass" , "cup" , "measuring_cup" , "water_bottle" , "coffee_mug" ,"coffeepot" , "pill_bottle" ,"pop_bottle" ,"wine_bottle" ,"beer_bottle"]
l3= ["cassette" , "cellular_telephone" , "wallet" , "iPod" , "notebook" , "bottlecap" , "remote_control" , "rubber_eraser" , "digital_watch"]
l4=[ "pencil_box" , "plate" , "toilet_tissue" , "baseball" , "croquet_ball" , "golf_ball" , "ping-pong_ball" , "tennis_ball" , "cheeseburger" ]

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.label = ''
        self.frame_to_predict = None
        self.classification = True
        self.score = .0
        ###
        self.q = queue.Queue()
        self.stage = 0
        self.corrections = 0
        self.all_grasps = [1, 2, 3, 4]
        self.Choose_grasp = list( self.all_grasps )
        self.grasp1 = None
        self.grasp_number =0
        self.grasp_name ="None"
        ###
        print( 'Loading network...' )
        #self.model = VGG16(weights='imagenet')
        self.model = ResNet50( weights='imagenet' )
        self.graph = tf.get_default_graph()
        print( 'Network loaded successfully!' )
        
    def run(self):
        
        with self.graph.as_default():
        
            while self.classification is True:
                if self.frame_to_predict is not None:
                    self.frame_to_predict = cv2.cvtColor(self.frame_to_predict, cv2.COLOR_BGR2RGB).astype(np.float32)
                    self.frame_to_predict = self.frame_to_predict.reshape((1, ) + self.frame_to_predict.shape)
                    self.frame_to_predict = imagenet_utils.preprocess_input(self.frame_to_predict)
                    predictions = self.model.predict(self.frame_to_predict)
                    (self.imageID, self.label, self.score) = imagenet_utils.decode_predictions(predictions)[0][0]
                    self.grasp_type()
                    #print ((self.label ,self.score))
                if self.classification == False :
                    break;

    def close(self):
        self.classification = False
        self.video_capture.release()
        cv2.destroyAllWindows()

    def grasp_type(self):
        if self.label in l1 :
            self.grasp_number =1
            self.grasp_name = "Pinch"
            print( ("Grasp_Type : Pinch \n ") )

        elif self.label in   l2:
            self.grasp_number =2
            self.grasp_name = "Palmar Wrist Neutral"
            print( ("Grasp_Type  : Palmar Wrist Neutral \n ") )

        elif self.label in l3:
            self.grasp_number =3
            self.grasp_name = "Tripod"
            print( ("Grasp_Type  : Tripod \n ") )

        elif self.label in l4:
            self.grasp_number =4
            self.grasp_name = "Palmar Wrist Pronated"
            print( ("Grasp_Type : Palmar Wrist Pronated \n ") )
        else :
            print (("Not Defined Grasp"))
            self.grasp_number =0
            self.grasp_name ="None"
        return self.grasp_number ,self.grasp_name


    def Main_algorithm(self):

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
            self.grasp1,_ = self.grasp_type( )

            print ((self.grasp1))
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

# Start a keras thread which will classify the frame returned by openCV
#keras_thread = MyThread()
#keras_thread.start()
#keras_thread.run_camera()
#time.sleep(2)
#keras_thread.close()


        

