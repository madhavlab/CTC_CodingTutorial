


import os 
import numpy as np

import pickle
import librosa
from sklearn.model_selection import train_test_split
import random



def data_gen(file_name):

      '''  inp: waves_yesno 
           out: [(spectogram, label_seq)]'''
            

      dat_dir = './'+file_name
      
      X = []
      for files in os.listdir(dat_dir):
           if '.wav' in files:

                wav = dat_dir + '/' + files #path to wav file
                
                wav, sr = librosa.load(wav, sr = 8000, mono= True) 
                   
                # loads data to generate 1d audio sequence 
                # librosa.load() returns (1d audio seq , sampling rate)
 
                
                #print(wav.shape)

                spec = np.abs(librosa.stft(wav, n_fft = 1024)) # each stft vector summarises 31 ms of audio 
                
                spec = spec.T # shape: (feats, TimeSize) -> (TimeSize, feats)

                #print(spec.shape)



                labs = np.array( [int(i) for i in files[:-4].split('_')])
                
                #print(labs)
                
                X.append((spec,labs))    



      return X





if __name__ == "__main__":

   X = data_gen('waves_yesno')

   X_train = X[:30]
   X_valid = X[30:45]
   X_test = X[45:]

   data = [X_train, X_valid, X_test]
   with open(f'./DATASET.bin', 'wb') as fp:
              pickle.dump(data, fp)
    
   print('DATA SAVED')


   
  
   print(len(X), X[0][0].shape, X[0][1])  
