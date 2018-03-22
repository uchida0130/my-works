#!/usr/bin/env python
import glob
import argparse
import numpy as np
from scipy.io.wavfile import read,write
import time
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import FFTanalysis
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
# import h5py
import erbscale
import separation
import eval
import csv

parser = argparse.ArgumentParser(description='DNN based Wiener Filter Estimation')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

pretrain_epoch = 10
n_epoch = 20
Fs = 16000
HFFTL = 513
DIM = [128]

def main(dim):
    # Prepare multi-layer perceptron model
    print "using GPU number: {}".format(args.gpu)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()

    #initialize parameter matrix
    l1_W = []
    l2_W = []
    l3_W = []
    l1b_W = []
    l2b_W = []
    n_units = 4*(dim*2+1)
    initializer = chainer.initializers.Normal()

    modelL1 = FunctionSet(l1=F.Linear(2*(dim*2+1), n_units, initialW = initializer),
                          l2=F.Linear(n_units, 2*(dim*2+1), initialW = initializer))
    # Setup optimizer
    optL1 = optimizers.Adam()
    optL1.setup(modelL1)
    modelL1.to_gpu()
    # Neural net architecture
    def pretrain_L1(x_data, ratio = 0.5, train=True):
        x = Variable(x_data)
        h1 = F.dropout(F.sigmoid(modelL1.l1(x)), ratio = ratio, train=train)
        y = F.dropout(F.sigmoid(modelL1.l2(h1)),ratio = ratio, train=train)
        return F.mean_squared_error(y, x), h1.data

    modelL2 = FunctionSet(l1=F.Linear(n_units, n_units, initialW = initializer),
                          l2=F.Linear(n_units, n_units, initialW = initializer))
    # Setup optimizer
    optL2 = optimizers.Adam()
    optL2.setup(modelL2)
    modelL2.to_gpu()
    # Neural net architecture
    def pretrain_L2(x_data, ratio = 0.5, train=True):
        x = Variable(x_data)
        h1 = F.dropout(F.sigmoid(modelL2.l1(x)), ratio = ratio, train=train)
        y = F.dropout(F.sigmoid(modelL2.l2(h1)),ratio = ratio, train=train)
        return F.mean_squared_error(y, x), h1.data

    # h5dataset = h5py.File("dataset.h5","w")
    dir = '/mnt/aoni02/uchida/work/selectfrq_estimated_data/'
    # h5dataset.create_group(dir)
    learned_data = 0
    pretrainsize = 2700
    learnsize = 2700
    testsize = 100
    HFFTL = 513
    m20dB_signal=sorted(glob.glob('-20dB_estimated_data/safia1_*.wav'))
    m20dB_noise =sorted(glob.glob('-20dB_estimated_data/safia2_*.wav'))
    m10dB_signal=sorted(glob.glob('-10dB_estimated_data/safia1_*.wav'))
    m10dB_noise =sorted(glob.glob('-10dB_estimated_data/safia2_*.wav'))
    m5dB_signal=sorted(glob.glob('-5dB_estimated_data/safia1_*.wav'))
    m5dB_noise =sorted(glob.glob('-5dB_estimated_data/safia2_*.wav'))
    zerosignal=sorted(glob.glob('estimated_data/safia1_*.wav'))
    zeronoise =sorted(glob.glob('estimated_data/safia2_*.wav'))
    p5dB_signal=sorted(glob.glob('5dB_estimated_data/safia1_*.wav'))
    p5dB_noise =sorted(glob.glob('5dB_estimated_data/safia2_*.wav'))
    p10dB_signal=sorted(glob.glob('10dB_estimated_data/safia1_*.wav'))
    p10dB_noise =sorted(glob.glob('10dB_estimated_data/safia2_*.wav'))

    estimated_signal = [[]]
    estimated_noise = [[]]
    #learning_dataset
    estimated_signal[0:learnsize/5] = m10dB_signal[0:learnsize/5]
    estimated_signal[learnsize/5:2*learnsize/5] = m5dB_signal[learnsize/5:2*learnsize/5]
    estimated_signal[2*learnsize/5:3*learnsize/5] = zerosignal[2*learnsize/5:3*learnsize/5]
    estimated_signal[3*learnsize/5:4*learnsize/5] = p5dB_signal[3*learnsize/5:4*learnsize/5]
    estimated_signal[4*learnsize/5:5*learnsize/5] = p10dB_signal[4*learnsize/5:5*learnsize/5]
    #test_dataset
    # estimated_signal[learnsize:learnsize+testsize] = m10dB_signal[0:testsize]
    # estimated_signal[learnsize+testsize:learnsize+2*testsize] = m5dB_signal[testsize:testsize*2]
    # estimated_signal[learnsize+2*testsize:learnsize+3*testsize] = zerosignal[testsize*2:testsize*3]
    # estimated_signal[learnsize+3*testsize:learnsize+4*testsize] = p5dB_signal[testsize*3:testsize*4]
    # estimated_signal[learnsize+4*testsize:learnsize+5*testsize] = p10dB_signal[testsize*4:testsize*5]
    estimated_signal[learnsize:learnsize+testsize] = m10dB_signal[learnsize:learnsize+testsize]
    estimated_signal[learnsize+testsize:learnsize+2*testsize] = m5dB_signal[learnsize:learnsize+testsize]
    estimated_signal[learnsize+2*testsize:learnsize+3*testsize] = zerosignal[learnsize:learnsize+testsize]
    estimated_signal[learnsize+3*testsize:learnsize+4*testsize] = p5dB_signal[learnsize:learnsize+testsize]
    estimated_signal[learnsize+4*testsize:learnsize+5*testsize] = p10dB_signal[learnsize:learnsize+testsize]
    estimated_signal[learnsize+5*testsize:learnsize+6*testsize] = m20dB_signal[learnsize:learnsize+testsize]

    estimated_noise[0:learnsize/5] = m10dB_noise[0:learnsize/5]
    estimated_noise[learnsize/5:2*learnsize/5] = m5dB_noise[learnsize/5:2*learnsize/5]
    estimated_noise[2*learnsize/5:3*learnsize/5] = zeronoise[2*learnsize/5:3*learnsize/5]
    estimated_noise[3*learnsize/5:4*learnsize/5] = p5dB_noise[3*learnsize/5:4*learnsize/5]
    estimated_noise[4*learnsize/5:5*learnsize/5] = p10dB_noise[4*learnsize/5:5*learnsize/5]

    # estimated_noise[learnsize:learnsize+testsize] = m10dB_noise[0:testsize]
    # estimated_noise[learnsize+testsize:learnsize+2*testsize] = m5dB_noise[testsize:testsize*2]
    # estimated_noise[learnsize+2*testsize:learnsize+3*testsize] = zeronoise[testsize*2:testsize*3]
    # estimated_noise[learnsize+3*testsize:learnsize+4*testsize] = p5dB_noise[testsize*3:testsize*4]
    # estimated_noise[learnsize+4*testsize:learnsize+5*testsize] = p10dB_noise[testsize*4:testsize*5]
    estimated_noise[learnsize:learnsize+testsize] = m10dB_noise[learnsize:learnsize+testsize]
    estimated_noise[learnsize+testsize:learnsize+2*testsize] = m5dB_noise[learnsize:learnsize+testsize]
    estimated_noise[learnsize+2*testsize:learnsize+3*testsize] = zeronoise[learnsize:learnsize+testsize]
    estimated_noise[learnsize+3*testsize:learnsize+4*testsize] = p5dB_noise[learnsize:learnsize+testsize]
    estimated_noise[learnsize+4*testsize:learnsize+5*testsize] = p10dB_noise[learnsize:learnsize+testsize]
    estimated_noise[learnsize+5*testsize:learnsize+6*testsize] = m20dB_noise[learnsize:learnsize+testsize]

    teacher_signal  =sorted(glob.glob("/mnt/aoni02/uchida/work/origin_data/teacher/*.wav"))

    def DNNbasedWienerfilter():

        #pretrain loop
        startexec = time.time()
        for epoch in xrange(pretrain_epoch):
            print "now proc: pretraining epoch{}".format(epoch)
            startepoch = time.time()
            perm = np.random.permutation(learnsize)
            for idx in np.arange(0,pretrainsize,3): #utterance Number training dataset
                # start = time.time()
                x_batch = np.empty((0,HFFTL*2),float)
                for iter in xrange(3):
                    fs,signal_data = read(estimated_signal[perm[idx+iter]] , "r")
                    fs,noise_data  = read(estimated_noise[perm[idx+iter]] , "r")
                    signal_data = signal_data/np.sqrt(np.mean(signal_data**2))
                    noise_data = noise_data/np.sqrt(np.mean(noise_data**2))
                    #FFT
                    Sspectrum_, synparam = FFTanalysis.FFTanalysis(signal_data)
                    Nspectrum_, synparam = FFTanalysis.FFTanalysis(noise_data)
                    N_FRAMES = np.shape(Sspectrum_)[0]
                    HFFTL  = np.shape(Sspectrum_)[1]
                    x_data = np.zeros((N_FRAMES,HFFTL*2))
                    for nframe in xrange(N_FRAMES):
                        spectrum = np.append(Sspectrum_[nframe],Nspectrum_[nframe])
                        x_data[nframe] = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in spectrum] #DNN indata
                    if iter == 0:
                        x_batch = np.append(x_batch,x_data,axis=0)
                    else:
                        x_batch = np.vstack((x_batch,x_data))
                for frq in xrange(HFFTL):
                    x_frqbatch = np.zeros((np.shape(x_batch)[0],2*(dim*2+1)),float)
                    x_frqbatch[:,dim] = x_batch[:,frq]
                    x_frqbatch[:,dim*3+1] = x_batch[:,frq+HFFTL]
                    for j in np.arange(1,dim+1):
                        if (frq - j) >= 0:
                            x_frqbatch[:,dim - j] = x_batch[:,frq - j]
                            x_frqbatch[:,dim*3+1 - j] = x_batch[:,frq+HFFTL - j]
                        if ((HFFTL-1) - (j+frq)) >= 0:
                            x_frqbatch[:,dim + j] = x_batch[:,frq + j]
                            x_frqbatch[:,dim*3+1 + j] = x_batch[:,frq+HFFTL + j]

                    x_frqbatch = x_frqbatch.astype(np.float32)
                    if epoch != 0 or idx != 0: #except first batch
                        modelL1.l1.W.data = cuda.to_gpu(l1_W.pop(0))
                        modelL2.l1.W.data = cuda.to_gpu(l2_W.pop(0))
                        modelL1.l2.W.data = cuda.to_gpu(l1b_W.pop(0))
                        modelL2.l2.W.data = cuda.to_gpu(l2b_W.pop(0))
                    # training
                    if args.gpu >= 0:
                        x_frqbatch = cuda.to_gpu(x_frqbatch)
                    optL1.zero_grads()
                    loss, hidden = pretrain_L1(x_frqbatch,ratio = 0.5)
                    loss.backward()
                    optL1.update()
                    optL2.zero_grads()
                    loss, hidden = pretrain_L2(hidden,ratio = 0.5)
                    loss.backward()
                    optL2.update()
                    #model parameter saving
                    l1_W.append(cuda.to_cpu(modelL1.l1.W.data))
                    l2_W.append(cuda.to_cpu(modelL2.l1.W.data))
                    l1b_W.append(cuda.to_cpu(modelL1.l2.W.data))
                    l2b_W.append(cuda.to_cpu(modelL2.l2.W.data))
            print 'pretrain epoch time:{0}sec'.format(np.round(time.time()-startepoch,decimals=2))
        # learning loop
        model = FunctionSet(l1=F.Linear(2*(dim*2+1), n_units, initialW = initializer),
                            l2=F.Linear(n_units, n_units, initialW = initializer),
                            l3=F.Linear(n_units, 1, initialW = initializer))
        # Setup optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        model.to_gpu()
        # Neural net architecture
        def forward(x_data, y_data, ratio = 0.5, train=True):
            x, t = Variable(x_data), Variable(y_data)
            h1 = F.dropout(F.sigmoid(model.l1(x)), ratio = ratio, train=train)
            h2 = F.dropout(F.sigmoid(model.l2(h1)),ratio = ratio, train=train)
            y  = model.l3(h2)
            return F.mean_squared_error(y, t), y
        startexec = time.time()
        for epoch in xrange(n_epoch):
            print "now proc: learning epoch{}".format(epoch)
            startepoch = time.time()
            perm = np.random.permutation(learnsize)
            for idx in np.arange(0,learnsize,3): #utterance Number training dataset
                # start = time.time()
                x_batch = np.empty((0,HFFTL*2),float)
                y_batch = np.empty((0,HFFTL),float)
                for iter in xrange(3):
                    fs,signal_data = read(estimated_signal[perm[idx+iter]] , "r")
                    fs,noise_data  = read(estimated_noise[perm[idx+iter]] , "r")
                    fs,teacher_data= read(teacher_signal[perm[idx+iter]] , "r")
                    signal_data = signal_data/np.sqrt(np.mean(signal_data**2))
                    noise_data = noise_data/np.sqrt(np.mean(noise_data**2))
                    teacher_data = teacher_data/np.sqrt(np.mean(teacher_data**2))


                    #FFT
                    Sspectrum_, synparam = FFTanalysis.FFTanalysis(signal_data)
                    Nspectrum_, synparam = FFTanalysis.FFTanalysis(noise_data)
                    Tspectrum_, synparam = FFTanalysis.FFTanalysis(teacher_data)

                    N_FRAMES = np.shape(Sspectrum_)[0]
                    HFFTL  = np.shape(Sspectrum_)[1]
                    x_data = np.zeros((N_FRAMES,HFFTL*2))
                    y_data = np.zeros((N_FRAMES,HFFTL))
                    if epoch == 0:
                        learned_data += N_FRAMES

                    for nframe in xrange(N_FRAMES):
                        spectrum = np.append(Sspectrum_[nframe],Nspectrum_[nframe])
                        x_data[nframe] = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in spectrum] #DNN indata
                        #phaseSpectrum = [np.arctan2(c.imag, c.real) for c in spectrum]
                        Spower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Sspectrum_[nframe]])
                        Tpower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Tspectrum_[nframe]])
                        for i, x in enumerate(Spower):
                            if x == 0:
                                Spower[i] = 1e-10
                        y_data[nframe] = Tpower/Spower
                    if iter == 0:
                        x_batch = np.append(x_batch,x_data,axis=0)
                        y_batch = np.append(y_batch,y_data,axis=0)
                    else:
                        x_batch = np.vstack((x_batch,x_data))
                        y_batch = np.vstack((y_batch,y_data))


                for frq in xrange(HFFTL):
                    x_frqbatch = np.zeros((np.shape(x_batch)[0],2*(dim*2+1)),float)
                    x_frqbatch[:,dim] = x_batch[:,frq]
                    x_frqbatch[:,dim*3+1] = x_batch[:,frq+HFFTL]
                    for j in np.arange(1,dim+1):
                        if (frq - j) >= 0:
                            x_frqbatch[:,dim - j] = x_batch[:,frq - j]
                            x_frqbatch[:,dim*3+1 - j] = x_batch[:,frq+HFFTL - j]
                        if ((HFFTL-1) - (j+frq)) >= 0:
                            x_frqbatch[:,dim + j] = x_batch[:,frq + j]
                            x_frqbatch[:,dim*3+1 + j] = x_batch[:,frq+HFFTL + j]
                    y_frqbatch = np.zeros((np.shape(y_batch)[0],1),float)
                    y_frqbatch = y_batch[:,frq].reshape(np.shape(y_batch)[0],1)

                    x_frqbatch = x_frqbatch.astype(np.float32)
                    y_frqbatch = y_frqbatch.astype(np.float32)

                    model.l1.W.data = cuda.to_gpu(l1_W.pop(0))
                    model.l2.W.data = cuda.to_gpu(l2_W.pop(0))
                    if epoch != 0 or idx != 0: #except first batch
                        model.l3.W.data = cuda.to_gpu(l3_W.pop(0))
                    # training
                    if args.gpu >= 0:
                        x_frqbatch = cuda.to_gpu(x_frqbatch)
                        y_frqbatch = cuda.to_gpu(y_frqbatch)
                    optimizer.zero_grads()
                    loss, pred = forward(x_frqbatch,y_frqbatch,ratio = 0.5)
                    loss.backward()
                    optimizer.update()
                    #model parameter saving
                    l1_W.append(cuda.to_cpu(model.l1.W.data))
                    l2_W.append(cuda.to_cpu(model.l2.W.data))
                    l3_W.append(cuda.to_cpu(model.l3.W.data))
            print 'epoch time:{0}sec'.format(np.round(time.time()-startepoch,decimals=2))
        f = open('selectfrq_estimated_data/pretrain_write1.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(l1_W)
        f.close()
        f = open('selectfrq_estimated_data/pretrain_write2.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(l2_W)
        f.close()
        f = open('selectfrq_estimated_data/pretrain_write3.csv', 'w')
        writer = csv.writer(f)
        writer.writerows(l3_W)
        f.close()
        #test loop
        SNRList = ["-10dB","-5dB","0dB","5dB","10dB","-20dB"]
        for SNRnum, SNR in enumerate(SNRList): #-10,-5,0,5,10,-20dB
            loss_sum = np.zeros(testsize)
            for idx in np.arange(learnsize+SNRnum*testsize,learnsize+(SNRnum+1)*testsize):
                fs,signal_data = read(estimated_signal[idx] , "r")
                fs,noise_data  = read(estimated_noise[idx] , "r")
                fs,teacher_data= read(teacher_signal[idx-testsize*SNRnum] , "r")
                signal_data = signal_data/np.sqrt(np.mean(signal_data**2))
                noise_data = noise_data/np.sqrt(np.mean(noise_data**2))
                teacher_data = teacher_data/np.sqrt(np.mean(teacher_data**2))

                Sspectrum_, synparam = FFTanalysis.FFTanalysis(signal_data)
                Nspectrum_, synparam = FFTanalysis.FFTanalysis(noise_data)
                Tspectrum_, synparam = FFTanalysis.FFTanalysis(teacher_data)

                N_FRAMES = np.shape(Sspectrum_)[0]
                HFFTL  = np.shape(Sspectrum_)[1]
                x_data = np.zeros((N_FRAMES,HFFTL*2))
                y_data = np.zeros((N_FRAMES,HFFTL))

                for nframe in xrange(N_FRAMES):
                    spectrum = np.append(Sspectrum_[nframe],Nspectrum_[nframe])
                    x_data[nframe] = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in spectrum] #DNN indata
                    #phaseSpectrum = [np.arctan2(c.imag, c.real) for c in spectrum]
                    Spower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Sspectrum_[nframe]])
                    Tpower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Tspectrum_[nframe]])
                    for i, x in enumerate(Spower):
                        if x == 0:
                            Spower[i] = 1e-10
                    y_data[nframe] = Tpower/Spower

                calcSNR = np.empty((N_FRAMES,0),float)
                totalloss  = np.zeros(HFFTL,float)
                # testing
                for frq in xrange(HFFTL):
                    model.l1.W.data = cuda.to_gpu(l1_W[frq])
                    model.l2.W.data = cuda.to_gpu(l2_W[frq])
                    model.l3.W.data = cuda.to_gpu(l3_W[frq])
                    # testing
                    x_frqdata = np.zeros((np.shape(x_data)[0],2*(dim*2+1)),float)
                    x_frqdata[:,dim] = x_data[:,frq]
                    x_frqdata[:,dim*3+1] = x_data[:,frq+HFFTL]
                    for j in np.arange(1,dim+1):
                        if (frq - j) >= 0:
                            x_frqdata[:,dim - j] = x_data[:,frq - j]
                            x_frqdata[:,dim*3+1 - j] = x_data[:,frq+HFFTL - j]
                        if ((HFFTL-1) - (j+frq)) >= 0:
                            x_frqdata[:,dim + j] = x_data[:,frq + j]
                            x_frqdata[:,dim*3+1 + j] = x_data[:,frq+HFFTL + j]
                    y_frqdata = np.zeros((np.shape(y_data)[0],1),float)
                    y_frqdata = y_data[:,frq].reshape(np.shape(y_data)[0],1)

                    x_frqdata = x_frqdata.astype(np.float32)
                    y_frqdata = y_frqdata.astype(np.float32)
                    if args.gpu >= 0:
                        x_frqdata = cuda.to_gpu(x_frqdata)
                        y_frqdata = cuda.to_gpu(y_frqdata)
                    loss, pred = forward(x_frqdata,y_frqdata, train = False)
                    totalloss[frq] = cuda.to_cpu(loss.data)
                    pred = np.reshape(cuda.to_cpu(pred.data), (N_FRAMES,1))
                    calcSNR= np.append(calcSNR, pred, axis = 1)
                fs,teacher_data= read(teacher_signal[idx-testsize*SNRnum] , "r")
                if teacher_data.dtype == "int16":
                    teacher_data = teacher_data/norm
                y_out = Sspectrum_ * calcSNR
                wf_signal = FFTanalysis.Synth(y_out,synparam,BPFon=0)
                wf_signal = wf_signal*np.sqrt(np.mean(teacher_data**2)/np.mean(wf_signal**2))
                write(dir+SNR+"/dim{}_DNNbased_No{}.wav".format(dim,idx-testsize*SNRnum),Fs,wf_signal)

        print 'exec time:{0}sec'.format(np.round(time.time()-startexec,decimals=2))
        print "data: ",learned_data

    def load_model():
        # learning loop
        model = FunctionSet(l1=F.Linear(2*(dim*2+1), n_units, initialW = initializer),
                            l2=F.Linear(n_units, n_units, initialW = initializer),
                            l3=F.Linear(n_units, 1, initialW = initializer))
        # Setup optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model)
        model.to_gpu()
        # Neural net architecture
        def forward(x_data, y_data, ratio = 0.5, train=True):
            x, t = Variable(x_data), Variable(y_data)
            h1 = F.dropout(F.sigmoid(model.l1(x)), ratio = ratio, train=train)
            h2 = F.dropout(F.sigmoid(model.l2(h1)),ratio = ratio, train=train)
            y  = model.l3(h2)
            return F.mean_squared_error(y, t), y
        with open('selectfrq_estimated_data/pretrain_write1.csv', 'r') as csvfile:
            readfile = csv.reader(csvfile)
            for row in readfile:
                print len(row)
                # print row
                l1_W.append(row)
        with open('selectfrq_estimated_data/pretrain_write2.csv', "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # print row
                l2_W.append(row)
        with open('selectfrq_estimated_data/pretrain_write3.csv', "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                # print row
                l3_W.append(row)
        #test loop
        SNRList = ["-10dB","-5dB","0dB","5dB","10dB","-20dB"]
        for SNRnum, SNR in enumerate(SNRList): #-10,-5,0,5,10,-20dB
            loss_sum = np.zeros(testsize)
            for idx in np.arange(learnsize+SNRnum*testsize,learnsize+(SNRnum+1)*testsize):
                fs,signal_data = read(estimated_signal[idx] , "r")
                fs,noise_data  = read(estimated_noise[idx] , "r")
                fs,teacher_data= read(teacher_signal[idx-testsize*SNRnum] , "r")
                signal_data = signal_data/np.sqrt(np.mean(signal_data**2))
                noise_data = noise_data/np.sqrt(np.mean(noise_data**2))
                teacher_data = teacher_data/np.sqrt(np.mean(teacher_data**2))

                Sspectrum_, synparam = FFTanalysis.FFTanalysis(signal_data)
                Nspectrum_, synparam = FFTanalysis.FFTanalysis(noise_data)
                Tspectrum_, synparam = FFTanalysis.FFTanalysis(teacher_data)

                N_FRAMES = np.shape(Sspectrum_)[0]
                HFFTL  = np.shape(Sspectrum_)[1]
                x_data = np.zeros((N_FRAMES,HFFTL*2))
                y_data = np.zeros((N_FRAMES,HFFTL))

                for nframe in xrange(N_FRAMES):
                    spectrum = np.append(Sspectrum_[nframe],Nspectrum_[nframe])
                    x_data[nframe] = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in spectrum] #DNN indata
                    #phaseSpectrum = [np.arctan2(c.imag, c.real) for c in spectrum]
                    Spower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Sspectrum_[nframe]])
                    Tpower = np.array([np.sqrt(c.real ** 2 + c.imag ** 2) for c in Tspectrum_[nframe]])
                    for i, x in enumerate(Spower):
                        if x == 0:
                            Spower[i] = 1e-10
                    y_data[nframe] = Tpower/Spower

                calcSNR = np.empty((N_FRAMES,0),float)
                totalloss  = np.zeros(HFFTL,float)
                # testing
                for frq in xrange(HFFTL):
                    model.l1.W.data = cuda.to_gpu(l1_W[frq])
                    model.l2.W.data = cuda.to_gpu(l2_W[frq])
                    model.l3.W.data = cuda.to_gpu(l3_W[frq])
                    # testing
                    x_frqdata = np.zeros((np.shape(x_data)[0],2*(dim*2+1)),float)
                    x_frqdata[:,dim] = x_data[:,frq]
                    x_frqdata[:,dim*3+1] = x_data[:,frq+HFFTL]
                    for j in np.arange(1,dim+1):
                        if (frq - j) >= 0:
                            x_frqdata[:,dim - j] = x_data[:,frq - j]
                            x_frqdata[:,dim*3+1 - j] = x_data[:,frq+HFFTL - j]
                        if ((HFFTL-1) - (j+frq)) >= 0:
                            x_frqdata[:,dim + j] = x_data[:,frq + j]
                            x_frqdata[:,dim*3+1 + j] = x_data[:,frq+HFFTL + j]
                    y_frqdata = np.zeros((np.shape(y_data)[0],1),float)
                    y_frqdata = y_data[:,frq].reshape(np.shape(y_data)[0],1)

                    x_frqdata = x_frqdata.astype(np.float32)
                    y_frqdata = y_frqdata.astype(np.float32)
                    if args.gpu >= 0:
                        x_frqdata = cuda.to_gpu(x_frqdata)
                        y_frqdata = cuda.to_gpu(y_frqdata)
                    loss, pred = forward(x_frqdata,y_frqdata, train = False)
                    totalloss[frq] = cuda.to_cpu(loss.data)
                    pred = np.reshape(cuda.to_cpu(pred.data), (N_FRAMES,1))
                    calcSNR= np.append(calcSNR, pred, axis = 1)
                fs,teacher_data= read(teacher_signal[idx-testsize*SNRnum] , "r")
                if teacher_data.dtype == "int16":
                    teacher_data = teacher_data/norm
                y_out = Sspectrum_ * calcSNR
                wf_signal = FFTanalysis.Synth(y_out,synparam,BPFon=0)
                wf_signal = wf_signal*np.sqrt(np.mean(teacher_data**2)/np.mean(wf_signal**2))
                write(dir+SNR+"/dim{}_DNNbased_No{}.wav".format(dim,idx-testsize*SNRnum),Fs,wf_signal)

    def run():
        start = time.time()
        # DNNbasedWienerfilter()
        load_model()
        elasped_time = int(time.time()-start)
        sec = elasped_time % 60
        minute = elasped_time / 60
        print 'elasped time:{0}min {1}sec'.format(minute, sec)

    if __name__ == '__main__':
        run()

if __name__ == '__main__':
    for i,dim in enumerate(DIM):
        main(dim)
