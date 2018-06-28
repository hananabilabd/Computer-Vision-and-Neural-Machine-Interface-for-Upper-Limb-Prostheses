import RealTime
import poweroff
import threading
#EMG ############################################################################################################
## All you have to do is to call start_thread to start threading smoothly
self.Real = RealTime.RealTime()

# self.Real.set_GP_instance(self)
self.Power = poweroff.poweroff()
self.thread1 = None
self.thread2 = None
self.event_stop_thread1 = threading.Event()
self.event_stop_thread2 = threading.Event()


def start_thread1(self):
    self.Real.Flag_predict = True
    self.Real.b = np.empty( [0, 8] )
    self.event_stop_thread1.clear()
    self.thread1 = threading.Thread( target=self.loop1 )
    self.thread1.start()



def loop1(self):
    while not self.event_stop_thread1.is_set():
        # if not self.stop_threads.is_set():
        if self.Real.myo_device.services.waitForNotifications( 1 ):
            print(self.Real.predictions_array)
            if len( self.Real.predictions_array ) % 5 == 0:
                self.Real.predictions_array.pop( 0 )
        else:
            print("Waiting...")



def stop_thread1(self):
    self.event_stop_thread1.set()
    self.thread1.join()
    self.thread1 = None
    self.Real.b = np.empty( [0, 8] )
    self.Real.Flag_Predict = False



def disconnect_MYO(self):
    print ("attempting to Disconnect")
    self.Real.myo_device.services.vibrate( 1 )  # short vibration
    # btle.Peripheral.disconnect()
    self.Real.myo_device.services.disconnect_MYO()
    print ("Successfully Disconnected")

###########################################################################################################3
