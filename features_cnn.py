'''
Created on 2 Nov 2016

@author: jl10015
'''

import glob
import os
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle


def framing(filename):


    sound_clip,s = librosa.load(filename,sr=11025*2)
    S = librosa.feature.melspectrogram(y=sound_clip, sr=s, n_mels=28,fmin=200,fmax=10000)       
    #mel =librosa.feature.melspectrogram(y=sound_clip,sr=s,n_mels=28,fmin=500,fmax=10000)
    mel = librosa.decompose.nn_filter(S,aggregate=np.median,metric='cosine')
   
    #print mel.shape
    
    mel=np.resize(mel, (28,420))

    #print mel[:,1:10]
    
    
    '''print mel.shape
    print mfcc_delta.shape
    print mfcc_delta2.shape'''
    
    mel2=np.zeros((28,28))

    #print mel2[:,0].shape
    
    #mel2[:,0]=np.mean(mel[:,15*0:15*(0+1)], axis=1)
    
    
    for i in xrange(28):
        mel2[:,i]= np.mean(mel[:,15*i:15*(i+1)], axis=1)

    #print mel2
    mel2=mel2.reshape((1,28*28))

    #exit()
    #plt.imshow(mel,interpolation=None,aspect='auto')
    #plt.show()
    '''
    window_size= sound_clip.size / (2*nbrFrames)
    
    
    prefeat=np.zeros((2*nbrFrames,2*window_size ))
    
    for i in xrange(nbrFrames):
            prefeat[2*i]=sound_clip[window_size*i:window_size*(i+2)]
            prefeat[2*i+1]=sound_clip[window_size*(i+1):window_size*(i+3)]
            
    
    features=np.zeros((2*nbrFrames,120))        
    for i in xrange(2*nbrFrames):                
        mfccs = np.mean(librosa.feature.mfcc(y=prefeat[i], sr=s, n_mfcc=40).T,axis=0)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs,order=2)
        features[i]=np.concatenate((mfccs,mfcc_delta,mfcc_delta2),axis=0)
    '''
    
    #print mel2.shape
    f = open("fmelspecdenoise28281.txt", "a")
    np.savetxt(f, mel2)
    f.close()
   
    

       
    
    
if __name__=='__main__':
    
    #framing('/home/jl10015/workspace/Birdy/wav/0a42af88-f61a-4504-9ba2.wav')
    
    k=0 
    for fn in os.listdir('/home/jl10015/workspace/Birdy/wav'):
        
        if k <8000:
            framing('/home/jl10015/workspace/Birdy/wav/'+fn)          
            fn=fn[:-4]
            f1=open('inmelspecdenoise1.txt',"a")
            f1.write(fn+'\n')
            f1.close()
            k=k+1
            print k



    
    
    
    
    
    