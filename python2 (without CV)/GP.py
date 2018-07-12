import numpy as np
from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
from PyQt4.QtGui import *
from PyQt4.QtCore import QObject,pyqtSignal
import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import threading
import EMG
#import CV
import EMG_Model
import collections
#import Queue as queue ##If python 2
import queue  ##If python 3
import pandas as pd
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
        self.EMG_Modeling =EMG_Model.EMG_Model()
        #self.cv =CV.CV()

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
        self.pushButton_9.clicked.connect(self.file_save_csv)

        self.pushButton_11.clicked.connect(self.start_thread0)
        self.pushButton_12.clicked.connect(self.stop_thread0)
        self.pushButton_10.clicked.connect( self.saveEMGModel )
        self.pushButton_10.setStyleSheet( "background-color: red" )
        self.pushButton_13.clicked.connect( self.browseCSVEMGModel1 )
        self.pushButton_14.clicked.connect( self.browseCSVEMGModel2 )
        self.pushButton_15.clicked.connect( self.browseCSVEMGModel3 )
        self.pushButton_16.clicked.connect( self.browseCSVEMGModel4 )
        self.pushButton_21.clicked.connect( self.joinCSV1 )
        self.pushButton_22.clicked.connect( self.joinCSV2 )
        self.pushButton_23.clicked.connect( self.saveJoinCSV )
        self.pushButton_17.clicked.connect( self.browsePickleEMGModel1 )
        self.pushButton_18.clicked.connect( self.browsePickleEMGModel2 )
        self.pushButton_19.clicked.connect( self.browseCVModel )
        self.pushButton_20.clicked.connect( self.start_thread4 )
        self.pushButton_20.setStyleSheet( "background-color: green" )
        self.pushButton_24.clicked.connect( self.stop_thread4 )
        self.pushButton_24.setStyleSheet( "background-color: red" )
        self.pushButton_25.clicked.connect( QtCore.QCoreApplication.instance().quit )
        self.path1=self.path2=self.path3=self.path4=self.path5 =self.path6 = self.path7 =self.path8 =None

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
        self.label.setPalette(palette)



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
        self.cv.q.queue.clear()
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread2 = True
        self.thread2 = threading.Thread(target = self.loop2)
        self.thread2.start()
    def start_thread4(self):##System
        self.listen.EMG = np.empty( [0, 8] )
        self.cv.q.queue.clear()
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread4 = True
        self.thread4 = threading.Thread(target = self.loop4)
        self.thread4.start()
        


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
            #self.update_predict()
            c = self.listen.predict(path =self.path7)
            if not c == None :
                self.cv.q.put(int(c))
                print (self.cv.q.queue)
            time.sleep(0.05)



    def loop4(self):##System
        while self.flag_thread4 :
            #self.update_predict()
            c = self.listen.predict(path =self.path8)
            if not c == None :
                self.cv.q.put(int(c))
                print (self.cv.q.queue)
                self.cv.Main_algorithm(path1=self.path9)
            time.sleep(0.05)
                
    def stop_thread0(self):
        self.listen.hub.stop()
        self.flag_thread0 =False
        self.thread0 = None
        self.listen.EMG = np.empty( [0, 8] )

    def stop_thread1(self):##Graph1
        self.listen.hub.stop()
        self.flag_thread1 = False
        self.thread1 = None
        self.listen.EMG = np.empty( [0, 8] )

    def stop_thread2(self):##Predict
        self.listen.hub.stop()
        self.flag_thread2 = False
        #self.event_stop_thread2.set()
        #self.thread2.join()
        self.thread2 = None
        self.listen.EMG = np.empty( [0, 8] )

    def stop_thread4(self):
        self.listen.hub.stop()
        self.flag_thread4 = False
        self.thread4 = None
        self.listen.EMG = np.empty( [0, 8] )


        
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
        for i in range( 8 ):
            self.emgcurve[i].setData( self.listen.EMG[:,i] )
            app.processEvents()
            
        if self.listen.EMG.shape[0] >= 1000 :
            self.listen.EMG = np.delete(self.listen.EMG,slice(0, 20), axis=0)



    def file_save_csv(self):

        self.path = QtGui.QFileDialog.getSaveFileName(self, 'Save Point', "", '*.csv')
        print (" Path = %s" %self.path)
        self.records=int(self.lineEdit.text())
        self.listen.EMG = np.empty( [0, 8] )
        threading.Thread( target=lambda: self.listen.hub.run_forever( self.listen.on_event ) ).start()
        self.flag_thread3 = True
        self.thread3 = threading.Thread( target=self.save_loop )
        self.thread3.start()

        #file.close()

    def save_loop(self):
        while self.listen.EMG.shape[0] < self.records:
            print (self.listen.EMG.shape[0])
            time.sleep( 0.01 )
        self.listen.hub.stop()
        np.savetxt( str( self.path ), self.listen.EMG, delimiter=",", fmt='%10.5f' )
        self.listen.EMG = np.empty( [0, 8] )
        print ("saved Sucessfully at %s" % self.path)
        self.thread3 = None

    def browseCSVEMGModel1(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_2.setText( filepath)
        self.path1 = str( filepath )
        print (" Path = %s" % self.path1)
        #self.records = int( self.lineEdit.text() )
    def browseCSVEMGModel2(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_6.setText( filepath)
        self.path2 = str( filepath )
        print (" Path = %s" % self.path2)
    def browseCSVEMGModel3(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_7.setText( filepath)
        self.path3 = str( filepath )
        print (" Path = %s" % self.path3)
    def browseCSVEMGModel4(self):

        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_8.setText( filepath)
        self.path4 = str( filepath )
        print (" Path = %s" % self.path4)
    def saveEMGModel(self):
        if not self.path1 ==None and not self.path2 ==None  and not self.path3 ==None and not self.path4 ==None :
            filepath = QtGui.QFileDialog.getSaveFileName( self, 'Save Point', "", '*.pickle' )

            self.EMG_Modeling.all_steps(path1=self.path1,path2=self.path2,path3=self.path3,path4=self.path4,file_name=str(filepath))
            print (" Saved SuccessFully at = %s" % filepath)
    def  joinCSV1(self):

        self.path5 = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_9.setText( self.path5)
        print (" Path = %s" % self.path5)
    def  joinCSV2(self):

        self.path6 = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.csv' )
        self.lineEdit_10.setText( self.path6)
        print (" Path = %s" % self.path6)
    def saveJoinCSV(self):
        if not self.path5 ==None and not self.path6 ==None :
            filepath = QtGui.QFileDialog.getSaveFileName( self, 'Save Point', "", '*.csv' )

            a=pd.read_csv(  str( self.path5 ) ,  header=None,index_col =False )
            b = pd.read_csv( str( self.path6 ) , header=None ,index_col =False)
            c = pd.concat( [a, b] )
            c.to_csv( str(filepath),index=False, header=None )
            print (" Saved SuccessFully at = %s" % filepath)
    def browsePickleEMGModel1(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.pickle' )
        self.lineEdit_3.setText( filepath)
        self.path7 = str( filepath )
        print (" Path = %s" % self.path7)
    def browsePickleEMGModel2(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.pickle')
        self.lineEdit_4.setText( filepath)
        self.path8 = str( filepath )
        print (" Path = %s" % self.path8)
    def browseCVModel(self):
        filepath = QtGui.QFileDialog.getOpenFileName( self, 'Single File', "", '*.h5' )
        self.lineEdit_5.setText( filepath)
        self.path9 = str( filepath )
        print (" Path = %s" % self.path9)


if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())



