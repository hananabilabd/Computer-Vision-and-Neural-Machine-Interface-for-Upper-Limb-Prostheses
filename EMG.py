#import scipy.io as sio
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn import svm
#from scipy import stats
#from sklearn.linear_model import SGDClassifier
#import sys
#import collections
from sklearn.externals import joblib
from scipy.signal import butter,lfilter,filtfilt
import numpy as np
import pandas as pd
import myo
import threading
import time
#import GP

class Listener(myo.DeviceListener):

  def __init__(self, queue_size=8):
    self.lock = threading.Lock()
    myo.init()
    self.hub = myo.Hub()
    #self.emg_data_queue = collections.deque(maxlen=queue_size)
    self.EMG = np.empty([0, 8])
    self.predictions_array = []
    self.prediction_array = np.empty( [0] )
    self.emg_total = np.empty( [0, 8] )
    self.flag_Graph1 = None
    self.flag_Predict = 0
    # self.set_GP_instance(GP)

  def set_GP_instance(self, GP):
    self.GP = GP

  def on_connected(self, event):
    event.device.stream_emg(True)

  def on_emg(self, event):
    with self.lock:
      #print (event.emg)
      self.EMG = np.append( self.EMG, [event.emg], axis=0 )
      #print(self.EMG.shape)
      #print (self.emg_data_queue)
      #self.emg_data_queue.append((event.timestamp, event.emg))
      #self.emg_data_queue.append( event.emg )

  def get_emg_data(self):
    with self.lock:
      return list(self.emg_data_queue)

  def plot_main(self):
    #print (listener.get_emg_data())
    #emgs = np.array([x[1] for x in listener.get_emg_data()]).T
    emgs = np.array( [x[0] for x in self.get_emg_data()] )
    print((emgs.shape))

  def filteration(self, data, sample_rate=2000.0, cut_off=20.0, order=5, ftype='highpass'):
    nyq = .5 * sample_rate
    b, a = butter( order, cut_off / nyq, btype=ftype )
    d = lfilter( b, a, data, axis=0 )
    return pd.DataFrame( d )

  def MES_analysis_window(self, df, width, tau, win_num):
    df_2 = pd.DataFrame()
    start = win_num * tau
    end = start + width
    df_2 = df.iloc[start:end]
    return end, df_2

  def features_extraction(self, df, th=0):
    # F1 : mean absolute value (MAV)
    MAV = abs( df.mean( axis=0 ) )

    MAV = list( MAV )
    WL = []
    SSC = []
    ZC = []
    for col, series in df.items():
      # F2 : wave length (WL)
      s = abs( np.array( series.iloc[:-1] ) - np.array( series.iloc[1:] ) )
      WL_result = np.sum( s )
      WL.append( WL_result )

      # F3 : zero crossing(ZC)
      _1starray = np.array( series.iloc[:-1] )
      _2ndarray = np.array( series.iloc[1:] )
      ZC.append( ((_1starray * _2ndarray < 0) & (abs( _1starray - _2ndarray ) >= th)).sum() )

      # F4 : slope sign change(SSC)
      _1st = np.array( series.iloc[:-2] )
      _2nd = np.array( series.iloc[1:-1] )
      _3rd = np.array( series.iloc[2:] )
      SSC.append( ((((_2nd - _1st) * (_2nd - _3rd)) > 0) & (
        ((abs( _2nd - _1st )) >= th) | ((abs( _2nd - _3rd )) >= th))).sum() )

    features_array = np.array( [MAV, WL, ZC, SSC] ).T
    return features_array

  def get_predictors(self, emg, width=512, tau=128):

    x = [];
    end = 0;
    win_num = 0;
    while ((len( emg ) - end) >= width):
      end, window_df = self.MES_analysis_window( emg, width, tau, win_num )
      win_num = win_num + 1

      ff = self.features_extraction( window_df )
      x.append( ff )

    predictors_array = np.array( x )

    nsamples, nx, ny = predictors_array.shape
    predictors_array_2d = predictors_array.reshape( (nsamples, nx * ny) )

    return np.nan_to_num( predictors_array_2d )


  def predict(self , path):
    if self.emg_total.shape[0] >= 512:
      self.flag_Predict =1
      #print ("Hiiii")
      self.emg_total = np.append( self.emg_total, self.EMG[:128], axis=0 )
      self.EMG = self.EMG[128:]
      data = pd.DataFrame( self.emg_total )
      filtered_emg = self.filteration( data, sample_rate=200 )
      predictors_test = self.get_predictors( filtered_emg )
      self.emg_total = self.emg_total[128:]
      filename = path
      pickled_clf = joblib.load( filename )
      return pickled_clf.predict( predictors_test )
    else :
      n= self.EMG.shape[0]
      self.emg_total = np.append( self.emg_total, self.EMG[:n], axis=0 )
      self.EMG = self.EMG[n:]
      return np.array([])




if __name__ == '__main__':
  l = Listener(queue_size = 512)
  #l.main()
  try:
    threading.Thread(target=lambda: l.hub.run_forever(l.on_event)).start()
    while True:
      #pass
      time.sleep(5)
      l.hub.stop()
      #l.plot_main()

  finally:
    l.hub.stop()# Will also stop the thread

