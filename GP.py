import numpy as np
from matplotlib.pyplot import axvline, axhline
import matplotlib.pyplot as plt
from PyQt4.uic import loadUiType
from PyQt4 import QtCore, QtGui
import matplotlib.backends.backend_qt4agg
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar)
from PyQt4.QtGui import *
import serial  # import Serial Library
#from drawnow import *
import pyqtgraph as pg
import pyqtgraph
import random
import sys, time
import RealTime
import poweroff
Ui_MainWindow, QMainWindow = loadUiType('GP.ui')

class Main(QMainWindow, Ui_MainWindow):


    def __init__(self, parent=None):
        #pyqtgraph.setConfigOption('background', 'w')  # before loading widget
        super(Main, self).__init__()
        self.setupUi(self)
        self.Real = RealTime.RealTime()
        self.Poweroff=poweroff.poweroff
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


        self.lastUpdateTime = time.time()
        self.show()
        
        #self.pushButton.clicked.connect( self.Real.start_MYO())
        self.pushButton.clicked.connect( self.Poweroff.power_off())
        #self.pushButton_2.clicked.connect( self.Real.thread_new() )
        #self.pushButton.clicked.connect(self.browse_wav)
        #self.pushButton.clicked.connect(self.file_save_txt)


        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Foreground, QtCore.Qt.red)
        self.label.setPalette(palette)

        self.EMG1.plotItem.showGrid(True, True, 0.7)
        self.EMG2.plotItem.showGrid(True, True, 0.7)
        self.EMG3.plotItem.showGrid(True, True, 0.7)
        self.EMG4.plotItem.showGrid(True, True, 0.7)
        #self.grFFT.plotItem.setRange(yRange=[0, 1])

    def plotter(self):

        self.data = [0]
        self.curve = self.plot.getPlotItem().plot()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updater)
        self.timer.start(0)





    def updater(self):

        arduinoString = arduinoData.readline()  # read the line of text from the serial port
        dataArray = arduinoString.split(',')
        a = float(dataArray[0])
        f = float(dataArray[1])


        self.data.append(a)
        # self.data.append(self.data[-1] + 0.2 * (0.5 - random.random()))
        self.curve.setData(self.data)
        if (len(self.data)% 512 == 0) and (f==0 or f==4):
            pen1 = pyqtgraph.mkPen(color='b')
            pen2 = pyqtgraph.mkPen(color='r')
            fftx,fft,phase =getFFT(self.data,38000)
            self.grFFT.plot(fftx,fft, pen=pen1, clear=True)
            self.grPhase.plot(fftx, phase, pen=pen2, clear=True)
            self.data = [0]
        if (len(self.data) % 512 == 0) and (f == 1 or f ==2 or f==3 or f==5 or f==6 or f==7):
            pen1 = pyqtgraph.mkPen(color='b')
            pen2 = pyqtgraph.mkPen(color='r')
            fftx, fft, phase = getFFT(self.data, 38000)
            self.grFFT_2.plot(fftx, fft, pen=pen1, clear=True)
            self.grPhase_2.plot(fftx, phase, pen=pen2, clear=True)
            self.data = [0]





        #self.login_widget_1 = LoginWidget(self)
        #self.verticalLayout_3.addWidget(self.login_widget_1)



        #self.pushButton_4.clicked.connect(self.browse_wav)
        #self.pushButton_4.setStyleSheet("background-color: red")

    def update_plots(self):
        ctime = time.time()
        if (ctime - self.lastUpdateTime) >= self.refreshRate:
            for i in range( 8 ):
                self.emgcurve[i].setData( self.listener.emg.data[i, :] )
            #for i in range( 4 ):
                #self.oricurve[i].setData( self.listener.orientation.data[i, :] )
            #for i in range( 3 ):
                #self.acccurve[i].setData( self.listener.acc.data[i, :] )
            self.lastUpdateTime = ctime

            app.processEvents()








    def browse_wav(self):
        self.flag1=1

        filepath = QtGui.QFileDialog.getOpenFileName(self, 'Single File', "C:\Users\Hanna Nabil\Desktop",'*.wav')
        f= str(filepath)
        if f != "":
            spf = wave.open(f, 'r')
        import contextlib

        with contextlib.closing(wave.open(f, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            print "Duration is " , duration



    def file_save_txt(self):
        from math import sin, pi
        b=0
        x = [0] * 38000

        for i in range(38000):
            x[i] = sin(2 * pi * 10000 * i / 38000) + sin(2 * pi * 500 * i / 38000)
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save Point', "C:\Users\Hanna Nabil\Desktop", '*.txt')
        file = open(name, "w")

        #for i in range(0, self.signal.size):
            #file.write(str(self.signal[i]) + "\n")
        for i in range(0,len(x)):
            b ='{:.6f}'.format(x[i])
            file.write(str(b) + "\n")

        file.close()



if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui
    import numpy as np

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())



