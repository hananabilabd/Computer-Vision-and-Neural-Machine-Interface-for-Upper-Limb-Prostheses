import numpy as np
from matplotlib.pyplot import axvline, axhline
import matplotlib.pyplot as plt
from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
import matplotlib.backends.backend_qt4agg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar)
from PyQt4.QtGui import *
from PyQt4.QtCore import QObject,pyqtSignal
import serial  # import Serial Library
#from drawnow import *
import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import threading
import EMG
import collections


Ui_MainWindow, QMainWindow = loadUiType('GP.ui')

class XStream(QObject):
    _stdout = None
    _stderr = None

    messageWritten = pyqtSignal(str)

    def flush( self ):
        pass

    def fileno( self ):
        return -1

    def write( self, msg ):
        if ( not self.signalsBlocked() ):
            self.messageWritten.emit(unicode(msg))

    @staticmethod
    def stdout():
        if ( not XStream._stdout ):
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout

    @staticmethod
    def stderr():
        if ( not XStream._stderr ):
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr

class Main(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)

        self.listen=EMG.Listener()

        XStream.stdout().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stdout().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        XStream.stderr().messageWritten.connect( self.textBrowser.insertPlainText )
        XStream.stderr().messageWritten.connect( self.textBrowser.ensureCursorVisible )
        
        #self.emgplot = pg.PlotWidget( name='EMGplot' )
        self.emgplot.setRange( QtCore.QRectF( -50, -200, 1000, 1400 ) )
        self.emgplot.disableAutoRange()
        self.emgplot.setTitle( "EMG" )

        self.refreshRate = 0.05
        self.emgcurve = []
        for i in range( 8 ):
            c = self.emgplot.plot( pen=(i, 10) )
            c.setPos( 0, i * 150 )
            self.emgcurve.append( c )
        
        self.emgcurve0 = [self.EMG1,self.EMG2,self.EMG3,self.EMG4,self.EMG5\
                           ,self.EMG6,self.EMG7,self.EMG8]
        for i in range (8):
            self.emgcurve0[i].plotItem.showGrid(True, True, 0.7)
            #self.emgcurve0[i].plotItem.setRange(yRange=[0, 1])
           
       

        #self.show()
        
        #self.pushButton.clicked.connect(self.Real.start_MYO)
        self.pushButton_2.clicked.connect( self.start_thread2)#Start Predict
        self.pushButton_3.clicked.connect( self.stop_thread2) # Stop Predict
        #self.pushButton_4.clicked.connect( self.disconnect_MYO)
        #self.pushButton_5.clicked.connect(self.Power.power_off)
        self.pushButton_6.clicked.connect( self.clear_textBrowser )
        self.pushButton_7.clicked.connect( self.start_thread1 )
        self.pushButton_8.clicked.connect( self.stop_thread1 )
        #self.pushButton_9.clicked.connect(self.file_save_csv)
        #self.pushButton_10.clicked.connect(self.browse_pickle)
        self.pushButton_11.clicked.connect(self.start_thread0)
        self.pushButton_12.clicked.connect(self.stop_thread0)
        #self.pushButton_4.setStyleSheet("background-color: red")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
        self.label.setPalette(palette)

        self.flag_thread2 = None
        self.event_stop_thread0 = threading.Event()
        self.event_stop_thread3 = threading.Event()

    def start_thread0(self):##Graph0
        self.listen.EMG = np.empty( [0, 8] )
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread0 = True
        self.thread0 = threading.Thread(target = self.loop0)
        self.thread0.start()

    def start_thread1(self):##Graph1
        self.listen.EMG = np.empty( [0, 8] )
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread1 = True
        self.thread1 = threading.Thread(target = self.loop1)
        self.thread1.start()

    def start_thread2(self):##Predict
        self.listen.EMG = np.empty( [0, 8] )
        self.listen.predictions_array=[]
        #self.event_stop_thread2.clear()
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread2 = True
        self.thread2 = threading.Thread(target = self.loop2)
        self.thread2.start()
        


    def loop0(self):
        while self.flag_thread0:
            self.update_Graph0()
            time.sleep(0.09)

    def loop1(self):
        while self.flag_thread1:
            self.update_Graph1()
            time.sleep(0.05)
        
    def loop2(self):
        #while not self.event_stop_thread2.is_set():
        while self.flag_thread2 :
            self.update_predict()
            time.sleep(0.05)

    def loop4(self):
        while  self.listen.EMG.shape[0] < self.records:
            print (self.listen.EMG.shape[0])
            if self.Real.myo_device.services.waitForNotifications(1):
                continue
            
        np.savetxt(str(self.path)+".csv", self.listen.EMG, delimiter="," ,fmt='%10.5f')
        self.Real.b= np.empty([0,8])
        print ("saved Sucessfully at %s" % self.path)
        #self.stop_thread4()
                
    def stop_thread0(self):
        self.listen.hub.stop()
        self.flag_thread0 =False
        self.thread0 = None
        self.listen.EMG = np.empty( [0, 8] )

    def stop_thread1(self):
        self.listen.hub.stop()
        self.flag_thread1 = False
        self.thread1 = None
        self.listen.EMG = np.empty( [0, 8] )

    def stop_thread2(self):
        self.listen.hub.stop()
        self.flag_thread2 = False
        #self.event_stop_thread2.set()
        #self.thread2.join()
        self.thread2 = None
        self.listen.EMG = np.empty( [0, 8] )
    def stop_thread3(self):
        self.event_stop_thread3.set()
        self.thread3.join()
        self.thread3 = None
        self.Real.b = np.empty( [0, 8] )
        self.Real.Flag_Graph0= False
    def stop_thread4(self):
        self.event_stop_thread4.set()
        self.thread4.join()
        self.thread4 = None

        
    def clear_textBrowser(self):          
        self.textBrowser.clear()
        
    def disconnect_MYO(self):
        print ("attempting to Disconnect")
        self.Real.myo_device.services.vibrate( 1 )  # short vibration
        #btle.Peripheral.disconnect()
        self.Real.myo_device.services.disconnect_MYO()
        print ("Successfully Disconnected")


    def update_Graph0(self):
        for i in range( 8 ):
            self.emgcurve0[i].plot(pen=(i, 10)).setData( self.listen.EMG[:,i] )
      
        #self.EMG1.plot(self.Real.b[:,0], pen=pen1,clear=True)
        #self.EMG2.plot(self.Real.b[:,1], pen=pen2, clear=True)
        #app.processEvents()
        if self.listen.EMG.shape[0] >=150 :
            self.listen.EMG = np.delete(self.listen.EMG,slice(0,30), axis=0)
   
        

    def update_Graph1(self):
        #ctime = time.time()
        #if (ctime - self.lastUpdateTime) >= self.refreshRate:

        #emgs = np.array( [x[0] for x in list( self.emg_data_queue )] )
        for i in range( 8 ):
            self.emgcurve[i].setData( self.listen.EMG[:,i] )

            app.processEvents()
            
        if self.listen.EMG.shape[0] >= 1000 :
            self.listen.EMG = np.delete(self.listen.EMG,slice(0, 20), axis=0)

    def update_predict(self):
        if  self.listen.EMG.shape[0] % 512 ==0 and not self.listen.EMG.shape[0] ==0:
            #self.listen.prediction_array.append( self.listen.predict( self.listen.EMG ) )

            self.listen.prediction_array = np.append( self.listen.prediction_array, self.listen.predict( self.listen.EMG ), axis=0 )
            print (self.listen.prediction_array)
            #self.b= np.empty([0,8])


    def browse_pickle(self):
        self.flag1=1

        filepath = QtGui.QFileDialog.getOpenFileName(self, 'Single File', "",'*.pickle')
        f= str(filepath)
        if f != "":
            spf = wave.open(f, 'r')
        import contextlib

        with contextlib.closing(wave.open(f, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            print "Duration is " , duration



    def file_save_csv(self):
      

        self.path = QtGui.QFileDialog.getSaveFileName(self, 'Save Point', "", '*.csv')
        print (" Path = %s" %self.path)
        self.records=int(self.lineEdit.text())
        self.Real.b= np.empty([0,8])
        self.event_stop_thread4 = threading.Event()
        self.event_stop_thread4.clear()
        self.thread4 = threading.Thread(target = self.loop4)
        self.thread4.start()
        

        #file.close()



if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())



