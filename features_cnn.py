'''
Created on 2 Nov 2016

@author: jl10015
'''


import os
import librosa
import numpy as np

def framing(filename):


    sound_clip,s = librosa.load(filename,sr=11025*2)
    mel =librosa.feature.mfcc(sound_clip,n_mfcc=28)

    mel=np.resize(mel, (28,420))
    
    mel2=np.zeros((28,28))
    
    for i in xrange(28):
        mel2[:,i]= np.mean(mel[:,15*i:15*(i+1)], axis=1)


    mel2=mel2.reshape((1,28*28))

   
    f = open("fmfcc2828.txt", "a")
    np.savetxt(f, mel2)
    f.close()
   
    

       
    
    
if __name__=='__main__':
      
    
    k=0 
    for fn in os.listdir('/home/jl10015/workspace/Birdy/wav'):
 
            framing('/home/jl10015/workspace/Birdy/wav/'+fn)          
            fn=fn[:-4]
            f1=open('index.txt',"a")
            f1.write(fn+'\n')
            f1.close()
            k=k+1
            print k
            



    
    
    
    
    
    
