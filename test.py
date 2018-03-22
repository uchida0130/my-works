import numpy as np
import json
# import h5py
import scipy.fftpack as fft
from scipy.io.wavfile import read,write
import FFTanalysis


def mul_broadcast():
    a = np.arange(12).reshape((3,4))
    b = np.arange(4).reshape((1,4))
    print a*b

def iotest():
    with open('test.txt','w') as f:
        f.write("test")

def jsontest():
    data = {
        "name": "aaa",
        "age": 30
    }
    print(data)
    print(data["name"])

    jsonstring = json.dumps(data, ensure_ascii=False) #json string type
    print(jsonstring)

    with open('test.json',"w") as f:
        json.dump(data,f)

    with open('test.json',"r") as f:
        data = json.load(f)
        print(data) #u-prefix don't care
        print(data["name"])

def h5pytest():
    sampling_frequency = [10.,100.,1000.]
    t = np.zeros((3,100))
    sample = np.zeros((3,100))
    output_file = "random.h5"
    h5file = h5py.File(output_file,"w")
    for i in np.arange(3):
        t[i] = np.arange(0,100.,1)/sampling_frequency[i]
        sample[i] = np.random.normal(size=len(t[i]))
        if i==0:
            sample_fft = [np.fft.fft(sample[i])]
        else:
            sample_fft = np.r_[sample_fft,[np.fft.fft(sample[i])]]
        dir = 'frequency_' + str(np.int(sampling_frequency[i]))
        h5file.create_group(dir)
        h5file.create_dataset(dir+'/random_number',data = sample[i])
        h5file.create_dataset(dir+'/spectrum',data = sample_fft[i])
        h5file.flush()
    h5file.flush()
    h5file.close()

def h5pyread():
    input_file="random.h5"

    h5file = h5py.File(input_file,"r")
    for i in np.array([10.,100.,1000.]):
        folder = "frequency_" + str(np.int(i))
        random = h5file[folder + "/random_number"].value
        spectrum = h5file[folder + "/spectrum"].value

def invarraytest():
    A = np.array(np.arange(40).reshape((10,4)))
    B = [[1.,0.],[1.,0.],[0.,1.],[0.,1.]]
    print A
    I = np.dot(A,B)
    print I
    A_ = np.dot(I,(B/np.sum(B,axis=0)).T)
    print A_

def poptest():
    A = np.zeros((3,4,5)) #(4,5) * 3
    B = np.arange(60).reshape(3,4,5)
    print B
    print B[0,:,4]
    C = B[0,:,3].reshape(1,len(B[0,:,3]))
    D = B[0,:,4].reshape(1,len(B[0,:,4]))
    print np.append(C,D,axis = 0)

def xrangetest():
    for batchframe in xrange(0,10,3):
        for nframe in xrange(batchframe,batchframe+3):
            print nframe

def vartest():
    A = np.arange(60).reshape(6,10)
    print A
    print np.var(A,1)

def slicetest():
    A = np.arange(10)
    print A[-2:-10:-1]
    print A[:-0]
    print np.arange(-10,15,5)
    SNRList = ["-10dB","-5dB","0dB","5dB","10dB","-20dB"]
    print SNRList[0]
# [8 7 6 5 4 3 2 1]

def axistest():
    A = np.arange(9).reshape(3,3)
    print A
    #[[0,1,2]
    # [3,4,5]
    # [6,7,8]]
    print np.mean(A,axis=0) #[3,4,5]
    print np.mean(A,axis=1) #[1,4,7]

def npappendtest():
    A = np.empty((0,3),int)
    B = np.arange(3).reshape(1,3)
    C = np.arange(3).reshape(1,3)
    A = np.append(A,B,axis=0)
    A = np.append(A,C,axis=0)

    print A

def ffttest():
    fs,voice = read("/home/uchida/anaconda2/work/origin_data/front_No1.wav")
    voice = voice/32768.
    # print "{0:05.3f}".format(np.mean(voice ** 2))
    Spectrum, synp = FFTanalysis.FFTanalysis(voice)
    # print "{} {} {} {} {} {} {}".format(synp.Fs, synp.N_samples, synp.FRAME_SIZE, synp.FRAME_SHIFT, synp.N_FRAMES, synp.FFTL, synp.WINDOW)
    resyn = FFTanalysis.Synth(Spectrum, synp)
    print "{0:05.3f}".format(np.mean((voice - resyn) ** 2))



def plottest():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import glob
    import FFTanalysis

    fs, est = read("estimated_data/safia2_No2702.wav")
    teacher_signal  =sorted(glob.glob("/mnt/aoni02/uchida/work/origin_data/*/*.wav"))
    print teacher_signal[2702]
    est = est / np.mean(np.abs(est))
    Nspectrum_, synparam = FFTanalysis.FFTanalysis(est)
    N_FRAMES = np.shape(Nspectrum_)[0]
    HFFTL  = np.shape(Nspectrum_)[1]
    plt.plot(np.arange(HFFTL),np.abs(Nspectrum_[0]))
    plt.savefig("figures/noisedominant.png")

def readwritetest():
    import wave
    mixed = wave.open("ch1/mixed_data/noise0_imp0_No2910.wav","rb")
    est = wave.open("estimated_data/DNNbased_No2910.wav","rb")
    f = wave.open("estimated_data/mvOKI1_No2910.wav","rb")
    print mixed.getparams()
    print est.getparams()
    print f.getparams()
    dnn = wave.open("DNN.wav","wb")

def gainsort():
    DNN = "estimated_data/DNNbased_No2702.wav"
    safia = "estimated_data/safia1_No2702.wav"
    fs,D = read(DNN)
    fs,S = read(safia)
    norm = 32768.
    if D.dtype == "int16":
        D = D/norm
    if S.dtype == "int16":
        S = S/norm
    D = D*np.sqrt(np.mean(S**2)/np.mean(D**2))
    write(DNN, fs, D)

def noisemasktest():
    mask = "noisemask/maskNo0000.csv"
    Nmask = np.loadtxt(mask,delimiter=",").T
    print np.shape(Nmask)

def writetest():
    f = open("test.txt","w")
    f.write("1\n")
    f.write("2\n")
    f.close

    # f = open("test.txt","r")
    # fdata = f.read()
    # fdata = fdata.split("\n")
    # print fdata[0]
    # print fdata[1]
    # f.close

def strtest():
    num = [1,2,3,4]
    dir = "/mnt/aoni02/uchida/work/ch{}/mixed_data".format(num)
    print dir

def readtest():
    norm = 32768.
    cleandir = "teacher/No0004.wav"
    noisedir = "area_out/-15dB/sub1_No0004.wav"
    fs,teach_data = read(cleandir,"r")
    if teach_data.dtype == "int16":
        teach_data = teach_data/norm
    teacherP = np.mean(teach_data ** 2)
    fs,noisy_data = read(noisedir,"r")
    if noisy_data.dtype == "int16":
        noisy_data = noisy_data/norm
    noisyP = np.mean(noisy_data ** 2)
    noisy_data = noisy_data * np.sqrt(teacherP/noisyP)
    write("sound/-15dBsub1_No0004.wav",fs,noisy_data)


    # import glob
    # cleandir = "selectfrq_estimated_data/clean_trainset_wav_16k/"
    # wavlist = glob.glob(cleandir+"*.wav")
    # norm = 32768.
    # for wavfile in wavlist:
    #     fs,wav_data = read(wavfile,"r")
    #     if wav_data.dtype == "int32":
    #         wave = wav_data/norm
    #         write(wavfile,fs,wave)

def beamareafig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import integrate
    frq = [500,1000,2000,4000,8000]
    colors=["b","g","r","c","m"]
    fig,ax=plt.subplots(1)
    for j,f in enumerate(frq):
        ramda = 34000/f
        mic_d = 3
        theta = np.arange(101)*np.pi/100
        fai = 2*np.pi*3*np.cos(theta)/ramda
        direc_sub = np.zeros(101)
        direc_add = np.zeros(101)
        for i,direction in enumerate(fai):
            def sin_sub(x):
                return np.abs(np.sin(x)-np.sin(x+direction))
            def sin_add(x):
                return np.abs((np.sin(x)+np.sin(x+direction))/4)
            direc_sub[i] = integrate.quad(sin_sub,0,2*np.pi)[0]
            direc_add[i] = integrate.quad(sin_add,0,2*np.pi)[0]
        ax.plot(theta/np.pi*180,direc_sub/4,label="{}_sub".format(f),color = colors[j])
        ax.legend()
        ax.plot(theta/np.pi*180,direc_add/4,label="{}_add".format(f),color = colors[j])
        ax.legend()
    fig.savefig("figure/beam.png")

if __name__ == '__main__':
    beamareafig()
