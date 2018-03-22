import glob
import numpy as np
from scipy.io.wavfile import read,write
import FFTanalysis

def run():
    Fs=16000
    FRAME_SIZE=1024
    FRAME_SHIFT=256
    FFTL=1024
    SPLIT_SIZE=1
    norm = 32768.
    #read files
    ch1s = sorted(glob.glob("/home/uchida/anaconda2/work/ch1/mixed_data/*.wav"))
    ch2s = sorted(glob.glob("/home/uchida/anaconda2/work/ch2/mixed_data/*.wav"))
    ch3s = sorted(glob.glob("/home/uchida/anaconda2/work/ch3/mixed_data/*.wav"))
    ch4s = sorted(glob.glob("/home/uchida/anaconda2/work/ch4/mixed_data/*.wav"))

    for index, ch1 in enumerate(ch1s):
        if index == 1:
            break
        print "nowproc: {}".format(index+1)
        N_mics=4
        fs,ch1 = read(ch1, "r")
        print ch1
        fs,ch2 = read(ch2s[index], "r")
        print ch2
        fs,ch3 = read(ch3s[index], "r")
        print ch3
        fs,ch4 = read(ch4s[index], "r")
        print ch4
        # %%% --- FFT --- %%%

        X1,synparam = FFTanalysis.FFTanalysis(ch1)
        N_FRAMES = np.shape(X1)[0]
        HFFTL = np.shape(X1)[1]
        X = np.zeros((N_mics,N_FRAMES,HFFTL),dtype=complex) # ch,NFRAME,frq
        X[0] = X1
        X_,synparam = FFTanalysis.FFTanalysis(ch2)
        X[1] = X_
        X_,synparam = FFTanalysis.FFTanalysis(ch3)
        X[2] = X_
        X_,synparam = FFTanalysis.FFTanalysis(ch4)
        X[3] = X_

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # %%% --- Beamforming  --- %%%
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%

        Sub1 = (X[0] - X[1])
        Add1 = (X[0] + X[1])/2
        Sub2 = (X[2] - X[3])
        Add2 = (X[2] + X[3])/2
        # %%%%%%%%%%%%%%%%%%%%%%
        # %%% --- Safia  --- %%%
        # %%%%%%%%%%%%%%%%%%%%%%

        Mask11,Mask12 = MaskGen(Add1,Sub1)
        Mask21,Mask22 = MaskGen(Add2,Sub2)
        SMask = Mask11*Mask21
        NMask = 1 - SMask
        # print "{} {}".format(np.shape(SMask),np.shape(NMask))
        # %%%%%%%%%%%%%%%%%%%%
        # %%% --- MVDR --- %%%
        # %%%%%%%%%%%%%%%%%%%%

        N_mic=2
        N_src=2
        MicInt=0.03
        Angle=0
        SV=340
        Delay = MicInt*np.sin(Angle)/SV
        Y_mvdrS = np.zeros((N_FRAMES,HFFTL),dtype=complex)
        Y_mvdrD = np.zeros((N_FRAMES,HFFTL),dtype=complex)
        W1_ = np.zeros((N_src,N_mic,HFFTL),dtype=complex)
        for frq in xrange(HFFTL):
            X_frq = np.append(X[0,:,frq].reshape(1,N_FRAMES),X[1,:,frq].reshape(1,N_FRAMES),axis = 0)
            for nsrc in xrange(N_src):
                R_kn = np.zeros((N_mic,N_mic),dtype=complex)
                R_n  = np.zeros((N_mic,N_mic),dtype=complex)
                R1 = np.zeros((N_mic,N_mic),dtype=complex)
                vsample1 = 0
                for nframe in xrange(N_FRAMES):
                    x_ = X_frq[:,nframe].reshape(N_src,1)
                    # % spatial matrix
                    R_ = np.dot(x_,np.conj(x_.T))
                    # % Correlation matrix of speech and noise
                    if nsrc == 0:
                        R_kn_ = SMask[nframe,frq]*R_
                        # % Correlation matrix of noise
                        R_n_  = NMask[nframe,frq]*R_

                    else:
                        R_kn_ = NMask[nframe,frq]*R_
                        # % Correlation matrix of noise
                        R_n_  = SMask[nframe,frq]*R_

                    # % check eigen value
                    # print "{} {}".format(np.shape(R_kn_),np.shape(R_n_))
                    w, V = np.linalg.eig(R_kn_-R_n_)
                    if np.max(np.real(w)) > 0:
                        R_kn = R_kn+R_kn_
                        R_n  = R_n+R_n_
                        R1   = R1+R_
                        vsample1=vsample1+1

                # % Steering vector
                # %-----------MV+masking----------

                w, V = np.linalg.eig((R_kn-R_n)/vsample1)
                idx = np.argmax(np.real(w))
                h1 = V[:,idx]
                # % MVDR coefficient
                R1 = R1 / vsample1
                w1 = np.dot(np.linalg.inv(R1),h1)/np.dot(np.conj(h1.T),np.dot(np.linalg.inv(R1),h1))
                # % Separation matrix
                W1_[nsrc,:,frq]=np.conj(w1.T)

            # % Calc output

            # %-----------MV+masking----------
            Y_frq1=np.dot(W1_[:,:,frq],X_frq)
            Y_mvdrS[:,frq] = Y_frq1[0]
            Y_mvdrD[:,frq] = Y_frq1[1]

        # %%%%%%%%%%%%%%%%%%%%%
        # %%% --- Synth --- %%%
        # %%%%%%%%%%%%%%%%%%%%%

        # % MV+masking output
        y_out1 = FFTanalysis.Synth(Y_mvdrS,synparam,300,5500)
        write("/home/uchida/anaconda2/work/estimated_data/S_mvdr_No{0:03d}.wav".format(index+1), Fs, y_out1)
        y_out2 = FFTanalysis.Synth(Y_mvdrD,synparam,300,5500)
        write("/home/uchida/anaconda2/work/estimated_data/N_mvdr_No{0:03d}.wav".format(index+1), Fs, y_out2)

def MaskGen(ObsSpec1,ObsSpec2):

    # %%% --- Comparison --- %%%
    Pow1=ObsSpec1*np.conj(ObsSpec1)
    Pow2=ObsSpec2*np.conj(ObsSpec2)
    Mask1=Pow1>Pow2
    Mask2=Pow2>Pow1
    return Mask1,Mask2

if __name__ == '__main__':
    run()
