import numpy as np
import scipy.signal
from scipy.io.wavfile import read,write
import scipy.fftpack as fft
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random

def overlap_save(impulse_,source_):
    impulse_length = 1024
    WINDOW = np.hamming(2*impulse_length)
    padding_imp = np.zeros((2*impulse_length),float)
    padding_imp[impulse_length:2*impulse_length] = impulse_

    h_ = padding_imp*WINDOW
    H_ = np.fft.fft(h_, n = 2*impulse_length)

    s_ = source_*WINDOW
    S_ = np.fft.fft(s_, n = 2*impulse_length)

    Y_ = H_*S_
    WINDOW = np.hamming(impulse_length)
    y_ = np.real(np.fft.ifft(Y_, 2*impulse_length))
    convolved = y_[0:impulse_length]*WINDOW

    return convolved

def impulse_conv(source):
    impulse_length = 1024
    frame_shift = impulse_length/4
    holding = 20
    ch_number = 2
    Fs = 16000
    impfile = glob.glob("ch1/imp_data/*.wav")
    fs,impulse = read(impfile[0])
    if fs != Fs:
        impulse = scipy.signal.decimate(impulse,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
    imp_ch = np.zeros((2,len(impulse)),float)

    for ch_iter in xrange(ch_number):
        impfile = glob.glob("ch{}/imp_data/*.wav".format(ch_iter+1))
        fs,impulse = read(impfile[0])
        imp = impulse / np.mean(impulse**2)
        if fs != Fs:
            imp = scipy.signal.decimate(imp,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
        imp_ch[ch_iter] = imp
    peak = np.argmax(imp_ch[0])

    imp_ch_ = np.zeros((2,impulse_length),float)
    for ch_iter in xrange(ch_number):
        imp_ch_[ch_iter] = imp_ch[ch_iter][peak-holding:peak-holding+impulse_length]

    l = len(source)
    convolved_source = np.zeros((2,l),float)
    for ch_iter in xrange(ch_number):
        for s_iter in xrange((l-2*impulse_length)/frame_shift+1):
            source_ = np.zeros((2*impulse_length),float)
            if s_iter < 4:
                source_[impulse_length-frame_shift*s_iter:2*impulse_length] = source[0:impulse_length+frame_shift*s_iter]
            else:
                source_[0:2*impulse_length] = source[(s_iter)*frame_shift:(s_iter)*frame_shift+(2*impulse_length)]
            convolved_source[ch_iter][s_iter*frame_shift:(s_iter)*frame_shift+impulse_length] += overlap_save(imp_ch_[ch_iter],source_)
        if 0 != l%frame_shift:
            source_ = np.zeros((2*impulse_length),float)
            source_[0:impulse_length+(l-((l/frame_shift)*frame_shift))] = source[(l/frame_shift)*frame_shift-impulse_length:l]
            convolved_source[ch_iter][(l/frame_shift)*frame_shift:l] += overlap_save(imp_ch_[ch_iter],source_)[0:(l-((l/frame_shift)*frame_shift))]

    return convolved_source

def noise_add(noisefile,convolved_source,SNR,start,power):
    Fs = 16000
    convolved_source = convolved_source / np.mean(convolved_source**2)
    fs,noise = read(noisefile)
    noise = noise / np.mean(noise**2)
    if fs != Fs:
        noise = scipy.signal.decimate(noise,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
    noise = noise[start:start+len(convolved_source)]
    noised_source = convolved_source*np.sqrt(np.mean(noise**2)/np.mean(convolved_source**2)*(10.**(SNR/10.))) + noise
    noised_source = noised_source*np.sqrt(power/np.mean(noised_source**2))

    return noised_source

def testdata_gen(snr,noise_num):
    Fs = 16000
    norm = 32768.
    ch_number = 2
    noiselen = 1600000 #100s*16000Hz
    noise_list = [9,11,13,14,18,20,26,28,30,47]

    f = open("startpoint/noise{0:02d}/{1:d}dB_startpoint.txt".format(noise_list[noise_num], snr),"r")
    fdata = f.read()
    fdata = fdata.split("\n")
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    voice = voicefile[8195]
    fs,voicesig = read(voice)
    voicesig = voicesig/norm
    for ch_iter in xrange(ch_number):
        noisefile = glob.glob("ch{}/noise_data/*.wav".format(ch_iter+1))
        convolved_source = impulse_conv(voicesig)
        start = int(fdata[195])
        noise = noisefile[noise_num]
        print "nowproc: noise{0:02d}, SNR {1:d}dB, testdata...".format(noise_list[noise_num], snr)
        noised_source = noise_add(noise,convolved_source[ch_iter],snr,start,np.mean(voicesig**2))
        write("ch{0:d}/test_data/{1:d}dB_noise{2:02d}_os_test.wav".format(ch_iter+1,snr,noise_list[noise_num]),Fs,noised_source)
    f.close

def getteachdata():
    Fs = 16000
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    for i, voice in enumerate(voicefile[0:10000]):
        fs,voicesig = read(voice)
        print "nowproc: No{0:04d}".format(i)
        convolved_source = impulse_conv(voicesig)
        write("teacher/No{0:04d}.wav".format(i),Fs,convolved_source[0])

def samplefileread(snr,noise_num):
    Fs = 16000
    norm = 32768.
    ch_number = 2
    noiselen = 1600000 #100s*16000Hz
    noise_list = [9,11,13,14,18,20,26,28,30,47]
    f = open("startpoint/noise{0:02d}/{1:d}dB_startpoint.txt".format(noise_list[noise_num], snr),"w")
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    voice = voicefile[0]
    fs,voicesig = read(voice)
    Mpower = np.mean((voicesig/norm)**2)
    for ch_iter in xrange(ch_number):
        noisefile = glob.glob("ch{}/noise_data/*.wav".format(ch_iter+1))
        noise = noisefile[noise_num]
        for i, voice in enumerate(voicefile[noise_num*1000:(noise_num+1)*1000]):
            fs,voicesig = read(voice)
            convolved_source = impulse_conv(voicesig)
            start = random.randint(0,noiselen-len(voicesig))
            f.write("{}\n".format(start))
            print "nowproc: noise{0:02d}, SNR {1:d}dB, No{2:04d}".format(noise_list[noise_num], snr, noise_num*1000+i)
            Y = noise_add(noise,convolved_source[ch_iter],snr,start,Mpower)
            write("ch{0:d}/{1:d}dB_mixed_data/noise{2:02d}/No{3:04d}.wav".format(ch_iter+1,snr,noise_list[noise_num],noise_num*1000+i),Fs,Y)
    f.close

def main():
    # getteachdata()
    SNR = [-10,-5,0,5,10]
    for snr in SNR:
        for noise_num in np.arange(10):
            samplefileread(snr,noise_num)
            # testdata_gen(snr,noise_num) #startpoint read version

if __name__ == '__main__':
    main()
