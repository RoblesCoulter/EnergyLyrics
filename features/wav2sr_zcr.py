import numpy as np
from numpy.lib                    import stride_tricks

import os

# signal processing
from scipy.io                     import wavfile
from scipy                        import stats, signal
from scipy.fftpack                import fft

from scipy.signal                 import lfilter, hamming
from scipy.fftpack.realtransforms import dct
from scikits.talkbox              import segment_axis
from scikits.talkbox.features     import mfcc

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    #print win
    #print frames
    frames *= win
    
    return np.fft.rfft(frames)  

def spectral_rolloff(wavedata, window_size, sample_rate, k=0.85):
    
    # convert to frequency domain
    magnitude_spectrum = stft(wavedata, window_size)
    power_spectrum     = np.abs(magnitude_spectrum)**2
    timebins, freqbins = np.shape(magnitude_spectrum)
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,timebins - 1) * (timebins / float(sample_rate)))
    
    sr = []

    spectralSum    = np.sum(power_spectrum, axis=1)
    
    for t in range(timebins-1):
        
        # find frequency-bin indeces where the cummulative sum of all bins is higher
        # than k-percent of the sum of all bins. Lowest index = Rolloff
        sr_t = np.where(np.cumsum(power_spectrum[t,:]) >= k * spectralSum[t])[0][0]
        
        sr.append(sr_t)
        
    sr = np.asarray(sr).astype(float)
    
    # convert frequency-bin index to frequency in Hz
    sr = (sr / freqbins) * (sample_rate / 2.0)
    
    return sr, np.asarray(timestamps)

def zero_crossing_rate(wavedata, block_length, samplerate):
    
    # how many blocks have to be processed?
    num_blocks = int(np.ceil(len(wavedata)/block_length))
    
    # when do these blocks begin (time in seconds)?
    timestamps = (np.arange(0,num_blocks - 1) * (block_length / float(samplerate)))
    
    zcr = []
    
    for i in range(0,num_blocks-1):
        
        start = i * block_length
        stop  = np.min([(start + block_length - 1), len(wavedata)])
        
        zc = 0.5 * np.mean(np.abs(np.diff(np.sign(wavedata[start:stop]))))
        zcr.append(zc)
    
    return np.asarray(zcr), np.asarray(timestamps)

def compute(path,preemph = False):
    # Convert to wave data
    samplerate, wavedata = wavfile.read(path)

    if preemph:
        # Pre-emphasis factor (to take into account the -6dB/octave
        # rolloff of the radiation at the lips level)
        prefac  = 0.97

        wavedata = lfilter([1., -prefac], 1, wavedata)

    # Get spectral roll off
    sr, ts = spectral_rolloff(wavedata,
                        #100000,
                        1024, 
                        samplerate,
                        k=0.85)

    if sr.shape[0] == 0:
        print "error"
        return None
    else:
        '''if sr.shape[0] > 1:
            count2 = count2 + 1
            print sr.shape'''
        sr1 =  np.median(sr)
        sr2 =  np.average(sr)

            
    # Get the avg ZCR
    zcr, ts = zero_crossing_rate(wavedata, 2048, samplerate);
    #print zcr, ts
    #print zcr.shape
    #print c
    zcr = np.average(zcr)

    feats = np.array([zcr,sr1,sr2])
    return feats
    

def compute_all():
    #in_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/wav/wav/german'
    in_dir = '/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/wav/wav/carlos/casa_may_24_2016/'
    #out_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/vectors/german/sr_zcr_preemp'
    out_dir = '/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/carlos/casa_may_24_2016/sr_zcr'


    count = 0
    count2 = 0

    preemph = False

    for fname in os.listdir(in_dir):
        path = os.path.join(in_dir, fname)
        if "." not in fname[0]:

            feats = compute(path, preemph)

            if feats is None:
                break
            
            np.savetxt('%s/%s'%(out_dir,fname.replace("wav","txt")), feats, delimiter=',') 
            print feats
            
            count = count + 1

    print "Processed %d and big %d"%(count,count2)

#compute_all()
#path = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/live/utterance_last.wav'
#path = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/wav/wav/english/DC_a03.wav'
#feats = compute(path)
#print feats

'''samplerate, wavedata = wavfile.read(in_dir+"/JE_sa14.wav")
print samplerate

sr, ts = spectral_rolloff(wavedata, 
                           100000, 
                           samplerate,
                           k=0.85)

if sr.shape[0] == 0:
    print "error"
else:
    print np.median(sr)
    print np.average(sr)
print sr,ts
print sr.shape'''
'''
# calculate zero-crossing-rate
zcr, ts = zero_crossing_rate(wavedata, 2048, samplerate);
print zcr, ts
print zcr.shape
print np.average(zcr)'''
