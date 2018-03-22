import glob
import numpy as np
import scipy.signal
from scipy.io.wavfile import read,write
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import FFTanalysis
import overlap_save
import random
from joblib import Parallel, delayed

def noise_add(noisefile, noisenum, voice_No, ch_num, H, voicefile,SNR,start):
    Fs = 16000
    norm = 32768.
    # fs,imp = read(impfile)
    # imp = imp/norm
    # if fs != Fs:
    #     imp = scipy.signal.decimate(imp,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
    # H_,synparam = FFTanalysis.FFTanalysis(imp)
    # maxF = np.argmax(np.mean(H_*np.conj(H_),axis=1))
    # H = H_[maxF].reshape(1,np.shape(H_)[1])
    fs,voice = read(voicefile)
    voice = voice/norm
    X,synparamX = FFTanalysis.FFTanalysis(voice)
    X = X*H #impulse response convolution
    Y = FFTanalysis.Synth(X,synparamX,BPFon = 0)

    fs,noise = read(noisefile)
    noise = noise/norm
    if fs != Fs:
        noise = scipy.signal.decimate(noise,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
    noise = noise[start:start+len(Y)]
    Z = Y + noise*np.sqrt(np.mean(Y**2)/np.mean(noise**2)*(10.**(-SNR/10.)))

    Z = Z*np.sqrt(np.mean(voice**2)/np.mean(Y**2))
    noise_teacher = noise*np.sqrt(np.mean(voice**2)/np.mean(noise**2)*(10.**(-SNR/10.)))

    if ch_num == 0:
        write("noise_teacher/{0:d}dB/noise{1:02d}/No{2:04d}.wav".format(SNR,noisenum,voice_No),Fs,noise_teacher)
    return Z

def convolution(H, voicefile):
    Fs = 16000
    norm = 32768.
    fs,voice = read(voicefile)
    voice = voice/norm
    X,synparamX = FFTanalysis.FFTanalysis(voice)
    X = X*H #impulse response convolution
    Y = FFTanalysis.Synth(X,synparamX,BPFon = 0)

    Y = Y*np.sqrt(np.mean(voice**2)/np.mean(Y**2))
    return Y

def samplefileread(snr,noise_num):
    Fs = 16000
    ch_number = 2
    noiselen = 1600000 #100s*16000Hz
    noise_list = [9,11,13,14,18,20,26,28]
    f = open("startpoint/noise{0:02d}/{1:d}dB_startpoint.txt".format(noise_list[noise_num], snr),"w")
    # fdata = f.read()
    # fdata = fdata.split("\n")
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    for i, voice in enumerate(voicefile[noise_num*250:(noise_num+1)*250]):
        fs,voicesig = read(voice)
        start = random.randint(0,noiselen-len(voicesig))
        f.write("{}\n".format(start))
        for ch_iter in xrange(ch_number):
            noisefile = sorted(glob.glob("ch{}/noise_data/*.wav".format(ch_iter+1)))
            noise = noisefile[noise_num]
            H_ = getimpulse(ch_iter)
            # start = int(fdata[i])
            # print "nowproc: noise{0:02d}, SNR {1:d}dB, No{2:04d}".format(noise_list[noise_num], snr, noise_num*250+i)
            Y = noise_add(noise, noise_list[noise_num], noise_num*250+i, ch_iter, H_, voice,snr,start)
            write("ch{0:d}/{1:d}dB_mixed_data/noise{2:02d}/No{3:04d}.wav".format(ch_iter+1,snr,noise_list[noise_num],noise_num*250+i),Fs,Y)
    f.close

def getimpulse(channel_num):
    ch_number = 2
    Fs = 16000
    norm = 32768.
    impulse_length = 1024
    holding = 20
    impfile = glob.glob("ch1/imp_data/*.wav")
    fs,imp = read(impfile[0])
    imp = imp/norm
    if fs != Fs:
        imp = scipy.signal.decimate(imp,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
    imp_ch = np.zeros((ch_number,len(imp)))
    for ch_iter in xrange(ch_number):
        impfile = glob.glob("ch{}/imp_data/*.wav".format(ch_iter+1))
        fs,imp = read(impfile[0])
        imp = imp/norm
        if fs != Fs:
            imp = scipy.signal.decimate(imp,int(np.ceil(fs/Fs)),n=512,ftype = "fir",zero_phase=True)
        imp_ch[ch_iter] = imp
    peak = np.argmax(imp_ch[0])
    H_ = np.zeros((ch_number,1,513),complex)
    for ch_iter in xrange(ch_number):
        H_[ch_iter] = FFTanalysis.FFTanalysis(imp_ch[ch_iter][peak-holding:peak-holding+impulse_length])[0]
        plt.plot(np.arange(len(H_[ch_iter][0])),10*np.log10(np.abs(H_[ch_iter][0])),label="ch{}".format(ch_iter))
    plt.legend()
    plt.savefig("figure/imp.png")
    plt.clf()
    return H_[channel_num]

def getteachdata(num):
    Fs = 16000
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    H_ = getimpulse(0)
    print "nowproc: No{0:04d}".format(num)
    Y = convolution(H_,voicefile[num])
    write("teacher/No{0:04d}.wav".format(num),Fs,Y)

def imptest():
    Fs = 16000
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    impfile1   = glob.glob("ch1/imp_data/*.wav")
    impfile2   = glob.glob("ch2/imp_data/*.wav")
    _,H1 = convolution(impfile1[0],voicefile[0])
    plt.plot(np.arange(len(H1)),H1,label="ch1")
    plt.title(impfile1[0])
    _,H2 = convolution(impfile2[0],voicefile[0])
    plt.plot(np.arange(len(H2)),H2,label="ch2")
    plt.title(impfile2[0])
    plt.legend()
    plt.savefig("figure/imp.png")
    plt.clf()
    plt.plot(np.arange(len(H2)),abs(H2-H1))
    plt.savefig("figure/imp_sub.png")
    plt.clf()

def getnoisesample(length,index,SNR):
    Fs=16000
    noisefile = glob.glob("ch1/noise_data/*.wav")
    fs, noise = read(noisefile[0])
    # fs, noise = read(noisefile[index/300])
    spfile = "{}dB_startpoint.txt".format(SNR)
    f = open(spfile,"r")
    fdata = f.read()
    fdata = fdata.split("\n")
    start = int(fdata[index])
    # write("ch1/noise_data/sample.wav",fs,noise[0:160000])
    return noise[start:length+start]

def main():
    # getteachdata()
    result = Parallel(n_jobs=-1,verbose=5)([delayed(getteachdata)(num) for num in np.arange(10000)])
    # SNR = [-5,0,5,10]
    # result = Parallel(n_jobs=-1,verbose=5)([delayed(samplefileread)(snr, noise_num) for snr in SNR for noise_num in np.arange(8)])
    # SNR = [-10,-5,0,5,10,15]
    # result = Parallel(n_jobs=-1)([delayed(testdata_gen)(snr, noise_num) for snr in SNR for noise_num in np.arange(2)])


def testdata_gen(snr,noise_num):
    Fs = 16000
    ch_number = 2
    noiselen = 1600000 #100s*16000Hz
    noise_list = [30,47]
    f = open("startpoint/noise{0:02d}/{1:d}dB_startpoint.txt".format(noise_list[noise_num], snr),"w")
    # fdata = f.read()
    # fdata = fdata.split("\n")
    voicefile = sorted(glob.glob("../work/origin_data/*/*.wav"))
    for i, voice in enumerate(voicefile[8000+noise_num*100:8000+(noise_num+1)*100]):
        fs,voicesig = read(voice)
        start = random.randint(0,noiselen-len(voicesig))
        f.write("{}\n".format(start))
        for ch_iter in xrange(ch_number):
            noisefile = sorted(glob.glob("ch{}/noise_data/*.wav".format(ch_iter+1)))
            noise = noisefile[noise_num]
            H_ = getimpulse(ch_iter)
            # start = int(fdata[i])
            print "nowproc: noise{0:02d}, SNR {1:d}dB, No{2:04d}".format(noise_list[noise_num], snr, 8000+noise_num*100+i)
            Y = noise_add(noise, noise_list[noise_num], 8000+noise_num*100+i, ch_iter, H_, voice,snr,start)
            write("ch{0:d}/test_data/noise{1:02d}/{2:d}dB/No{3:04d}.wav".format(ch_iter+1,noise_list[noise_num],snr,8000+noise_num*100+i),Fs,Y)
    f.close

if __name__ == '__main__':
    main()
    # imptest()
    # getteachdata()
    # for index in np.arange(2200,2300):
    #     inputname = "ch2/-20dB_mixed_data/noise0_imp0_No{}.wav".format(index)
    #     fs, inputsignal = read(inputname)
    #     noise = getnoisesample(len(inputsignal),index,-20)
    #     write("matlab_output/noise/noise_No{}.wav".format(index),fs,noise)
