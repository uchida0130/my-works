import numpy as np
import FFTanalysis
import noising
from scipy.io.wavfile import read,write
import matplotlib
import matplotlib.pyplot as plt

def xcorr(x,y):
    center = len(x)
    acor = np.correlate(x,y,"full")
    lags = np.argmax(acor[len(acor)//2:])
    return acor, lags

def evaluation():
    norm = 32768.
    fs, noised_data = read("ch1/mixed_data/noise0_imp0_No0000.wav")
    fs, estimated = read("estimated_data/mvOKI1_No0000.wav")
    fs, noise = read("estimated_data/mvOKI2_No0000.wav")
    fs, reference = read("origin_data/f001/nf001001.wav")
    if noised_data.dtype == "int16":
        noised_data = noised_data / norm
    if estimated.dtype == "int16":
        estimated = estimated / norm
    if noise.dtype == "int16":
        noise = noise / norm
    if reference.dtype == "int16":
        reference = reference / norm
    # acor, lags = xcorr(reference,estimated)
    lags = 185
    reference[lags:] = reference[:-lags]
    reference[:lags] = 0
    reference = np.append(reference,noise,axis=0)
    estimated = np.append(estimated,noise,axis=0)
    sdr,sir,sar,perm = separation.bss_eval_sources(reference,estimated)
    print sdr
    noised_data = np.append(noised_data,noise,axis=0)
    sdr,sir,sar,perm = separation.bss_eval_sources(reference,noised_data)
    print sdr

def posteval():
    import noising
    import glob
    fs, safiaS = read("estimated_data/safia1_No2702.wav")
    fs, safiaN = read("estimated_data/safia2_No2702.wav")
    teacher_signal  =sorted(glob.glob("/mnt/aoni02/uchida/work/origin_data/teacher/*.wav"))
    fs,T = read(teacher_signal[2702])
    N = noising.getnoisesample(len(safiaS),2702)
    if T.dtype == "int16":
        T = T/norm
    signal_data = safiaS/np.mean(np.abs(safiaS))
    noise_data = safiaN/np.mean(np.abs(safiaN))
    teacher_data = T/np.mean(np.abs(T))
    tnoise_data = N/np.mean(np.abs(N))
    reference = np.vstack((teacher_data,tnoise_data))
    print reference.shape
    estimated = np.vstack((signal_data,noise_data))
    sdr,sir,sar,perm = separation.bss_eval_sources(reference,estimated)
    print "SDR = {}\nSIR = {}".format(sdr[0],sir[0])

    fs, mixed = read("ch1/mixed_data/noise0_imp0_No2702.wav")
    mixed_data = mixed/np.mean(np.abs(mixed))
    estimated = np.vstack((mixed_data,tnoise_data))
    sdr,sir,sar,perm = separation.bss_eval_sources(reference,estimated)
    print "SDR = {}\nSIR = {}".format(sdr[0],sir[0])

def NRReval(Ssignal,Nsignal,MaskedSsignal,MaskedNsignal):
    SNRin = 10*np.log10(np.mean(np.abs(Ssignal)**2) / np.mean(np.abs(Nsignal)**2))
    # print "SNRin={}".format(np.round(SNRin,decimals=2))
    SNRout= 10*np.log10(np.mean(np.abs(MaskedSsignal)**2) / np.mean(np.abs(MaskedNsignal)**2))
    return SNRout-SNRin,SNRout

def LSDeval(TrueSpec,EstSpec):
    TruePower = np.zeros((np.shape(EstSpec)[0],np.shape(EstSpec)[1]),float)
    EstPower = np.zeros((np.shape(EstSpec)[0],np.shape(EstSpec)[1]),float)
    for i,EstNFRAME in enumerate(EstSpec):
        for j,Estbin in enumerate(EstNFRAME):
            if Estbin == 0:
                EstPower[i][j] = 0
            else:
                EstPower[i][j] = 20*np.log10(np.abs(Estbin))
    for i,TrueNFRAME in enumerate(TrueSpec):
        for j,Truebin in enumerate(TrueNFRAME):
            if Truebin == 0:
                TruePower[i][j] = 0
            else:
                TruePower[i][j] = 20*np.log10(np.abs(Truebin))
    return np.mean(np.abs(TruePower - EstPower))

def NRR_LSDeval(inSpec,outSpec,targetSpec,noiseSpec,target,noise,synpS,synpN):
    # Spectrum each
    Mask = np.zeros((np.shape(inSpec)[0],np.shape(inSpec)[1]),float)
    for i,inNFRAME in enumerate(inSpec):
        for j,inbin in enumerate(inNFRAME):
            if inbin == 0:
                Mask[i][j] = 0
            else:
                Mask[i][j] = np.abs(outSpec[i][j])/np.abs(inSpec[i][j])
    MaskedSsignal = FFTanalysis.Synth(targetSpec*Mask,synpS,BPFon=0)
    MaskedNsignal = FFTanalysis.Synth(noiseSpec*Mask,synpN,BPFon=0)
    NRR,SNRout = NRReval(target,noise,MaskedSsignal,MaskedNsignal)
    LSD = LSDeval(targetSpec,outSpec)
    return NRR,LSD,SNRout

def evalmain(inputname,outputname,targetname,index,SNR):
    fs, inputsignal = read(inputname)
    fs, outputsignal = read(outputname)
    fs, targetsignal = read(targetname)
    noise = noising.getnoisesample(len(inputsignal),index,SNR)
    noise = noise*np.sqrt(np.mean(targetsignal**2)/np.mean(noise**2))*(10 **(-SNR/10.))
    #-5dB noise = noise * (10 **(-SNR/10.))

    inSpec, synp = FFTanalysis.FFTanalysis(inputsignal)
    outSpec, synp = FFTanalysis.FFTanalysis(outputsignal)
    targetSpec, synpS = FFTanalysis.FFTanalysis(targetsignal)
    noiseSpec, synpN = FFTanalysis.FFTanalysis(noise)

    NRR,LSD,SNRout = NRR_LSDeval(inSpec,outSpec,targetSpec,noiseSpec,targetsignal,noise,synpS,synpN)
    # print "NRR={}".format(np.round(NRR,decimals = 2))
    # print "LSD={}".format(np.round(LSD,decimals = 2))
    return NRR,LSD,SNRout

def eachSNReval():
    NRR_sum1 = 0
    LSD_sum1 = 0
    NRR_sum2 = 0
    LSD_sum2 = 0
    NRR_sum3 = 0
    LSD_sum3 = 0
    for i, SNR in enumerate(np.arange(-10,15,5)):
        NRR_1 = 0
        LSD_1 = 0
        NRR_2 = 0
        LSD_2 = 0
        NRR_3 = 0
        LSD_3 = 0
        for index in np.arange(20):
            # if (2100+index+20*i)%10 == 0:
                # print "now proc: No{}".format(2100+index+20*i)
            if SNR == 0:
                inputname = "estimated_data/safia1_No{}.wav".format(2100+index+20*i)
            elif SNR == 5:
                inputname = "5dB_estimated_data/safia1_No{}.wav".format(2100+index+20*i)
            elif SNR == 10:
                inputname = "10dB_estimated_data/safia1_No{}.wav".format(2100+index+20*i)
            elif SNR == -5:
                inputname = "-5dB_estimated_data/safia1_No{}.wav".format(2100+index+20*i)
            elif SNR == -10:
                inputname = "-10dB_estimated_data/safia1_No{}.wav".format(2100+index+20*i)

            outputname = "fullSNR_estimated_data/DNNbased_No{}.wav".format(2100+index+20*i)
            targetname = "origin_data/teacher/No{}.wav".format(2100+index+20*i)
            NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,2100+index+20*i,SNR)
            NRR_1 += NRR
            LSD_1 += LSD

            # #oracle
            # outputname = "oracle/fullSNR/oracle_No{}.wav".format(2100+index+20*i)
            # NRR,LSD = evalmain(inputname,outputname,targetname,2100+index+20*i,SNR)
            # NRR_2 += NRR
            # LSD_2 += LSD
            #
            # #zelinsky
            # outputname = "zelinsky/fullSNR/No{}.wav".format(2100+index+20*i)
            # NRR,LSD = evalmain(inputname,outputname,targetname,2100+index+20*i,SNR)
            # NRR_3 += NRR
            # LSD_3 += LSD

        print "SNR = {}".format(SNR)
        print "---proposed---\n"
        print "NRR={}".format(np.round(NRR_1/100.,decimals = 2))
        print "LSD={}\n".format(np.round(LSD_1/100.,decimals = 2))
        # print "---oracle---\n"
        # print "NRR={}".format(np.round(NRR_2/20.,decimals = 2))
        # print "LSD={}\n".format(np.round(LSD_2/20.,decimals = 2))
        # print "---zelinsky---\n"
        # print "NRR={}".format(np.round(NRR_3/20.,decimals = 2))
        # print "LSD={}\n".format(np.round(LSD_3/20.,decimals = 2))

        NRR_sum1 += NRR_1
        LSD_sum1 += LSD_1
        # NRR_sum2 += NRR_2
        # LSD_sum2 += LSD_2
        # NRR_sum3 += NRR_3
        # LSD_sum3 += LSD_3

    print "---proposed---\n"
    print "NRR={}".format(np.round(NRR_sum1/500.,decimals = 2))
    print "LSD={}\n".format(np.round(LSD_sum1/500.,decimals = 2))
    # print "---oracle---\n"
    # print "NRR={}".format(np.round(NRR_sum2/100.,decimals = 2))
    # print "LSD={}\n".format(np.round(LSD_sum2/100.,decimals = 2))
    # print "---zelinsky---\n"
    # print "NRR={}".format(np.round(NRR_sum3/100.,decimals = 2))
    # print "LSD={}\n".format(np.round(LSD_sum3/100.,decimals = 2))

def SEGANeval():
    NRR_sum1 = 0
    LSD_sum1 = 0
    for i, SNR in enumerate(np.arange(-10,15,5)):
        NRR_1 = 0
        LSD_1 = 0
        for index in np.arange(100):
            # if (2100+index+20*i)%10 == 0:
                # print "now proc: No{}".format(2100+index+20*i)
            if SNR == 0:
                inputname = "estimated_data/safia1_No{}.wav".format(2100+index)
                outputname = "segan/test_cleanwav/0dBNo{}.wav".format(2100+index)
            elif SNR == 5:
                inputname = "5dB_estimated_data/safia1_No{}.wav".format(2100+index)
                outputname = "segan/test_cleanwav/5dBNo{}.wav".format(2100+index)
            elif SNR == 10:
                inputname = "10dB_estimated_data/safia1_No{}.wav".format(2100+index)
                outputname = "segan/test_cleanwav/10dBNo{}.wav".format(2100+index)
            elif SNR == -5:
                inputname = "-5dB_estimated_data/safia1_No{}.wav".format(2100+index)
                outputname = "segan/test_cleanwav/m5dBNo{}.wav".format(2100+index)
            elif SNR == -10:
                inputname = "-10dB_estimated_data/safia1_No{}.wav".format(2100+index)
                outputname = "segan/test_cleanwav/m10dBNo{}.wav".format(2100+index)

            targetname = "origin_data/teacher/No{}.wav".format(2100+index)
            NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,2100+index,SNR)
            NRR_1 += NRR
            LSD_1 += LSD

        print "SNR = {}".format(SNR)
        print "---proposed---\n"
        print "NRR={}".format(np.round(NRR_1/100.,decimals = 2))
        print "LSD={}\n".format(np.round(LSD_1/100.,decimals = 2))

        NRR_sum1 += NRR_1
        LSD_sum1 += LSD_1

    print "---proposed---\n"
    print "NRR={}".format(np.round(NRR_sum1/500.,decimals = 2))
    print "LSD={}\n".format(np.round(LSD_sum1/500.,decimals = 2))

def oneSNReval(SNR):
    NRR_1 = 0
    LSD_1 = 0
    NRR_2 = 0
    LSD_2 = 0
    NRR_3 = 0
    LSD_3 = 0
    for index in np.arange(2700,2800):
        # if (index)%10 == 0:
        #     print "now proc: No{}".format(index)
        inputname = "{}dB_estimated_data/safia1_No{}.wav".format(SNR,index)
        if SNR == 0:
            inputname = "estimated_data/safia1_No{}.wav".format(index)
        outputname = "selectfrq_estimated_data/{}dB/dim128_DNNbased_No{}.wav".format(SNR,index)
        targetname = "origin_data/teacher/No{}.wav".format(index)
        NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,index,SNR)
        NRR_1 += NRR
        LSD_1 += LSD

        # #oracle
        # outputname = "oracle/{}dB/oracle_No{}.wav".format(SNR,index)
        # NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,index,SNR)
        # NRR_2 += NRR
        # LSD_2 += LSD
        #
        # #zelinsky
        # outputname = "zelinsky/{}dB/No{}.wav".format(SNR,index)
        # NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,index,SNR)
        # NRR_3 += NRR
        # LSD_3 += LSD

    # print "SNR = {}dB, dim = {}".format(SNR,dim)
    # print "---proposed---"
    # print "NRR={}".format(np.round(NRR_1/100.,decimals = 2))
    # print "LSD={}\n".format(np.round(LSD_1/100.,decimals = 2))
    # print "---oracle---\n"
    # print "NRR={}".format(np.round(NRR_2/100.,decimals = 2))
    # print "LSD={}\n".format(np.round(LSD_2/100.,decimals = 2))
    # print "---zelinsky---\n"
    # print "NRR={}".format(np.round(NRR_3/100.,decimals = 2))
    # print "LSD={}\n".format(np.round(LSD_3/100.,decimals = 2))
    return NRR_1/100,LSD_1/100

def DemoSignalNormalize(SNR,dim,index):
    inputname = "{}dB_estimated_data/safia1_No{}.wav".format(SNR,index)
    if SNR == 0:
        inputname = "estimated_data/safia1_No{}.wav".format(index)
    # outputname = "fullSNR_estimated_data/{}dB/dim{}_DNNbased_No{}.wav".format(SNR,dim,index)
    # outputname = "oracle/{}dB/oracle_No{}.wav".format(SNR,index)
    # outputname = "zelinsky/{}dB/No{}.wav".format(SNR,index)
    # outputname = "sound/{}dB/zeli_No{}.wav".format(SNR,index)
    outputname = "fullSNR_estimated_data/DNNbased_No{}.wav".format(index)
    targetname = "origin_data/teacher/No{}.wav".format(index)
    NRR,LSD,SNRout = evalmain(inputname,outputname,targetname,index,SNR)
    fs, inputsignal = read(inputname)
    fs, outputsignal = read(outputname)
    fs, targetsignal = read(targetname)
    # norm = 32768.
    # if inputsignal.dtype == "int16":
    #     inputsignal = inputsignal/norm
    # if outputsignal.dtype == "int16":
    #     outputsignal = outputsignal/norm
    # if targetsignal.dtype == "int16":
    #     targetsignal = targetsignal/norm
    # NormalizedSignal = inputsignal*np.sqrt(np.mean(outputsignal**2))/np.sqrt(np.mean(inputsignal**2))
    NormalizedSignal = outputsignal*np.sqrt(1 + 1/(10**(SNRout/10)))/np.sqrt(10**(-SNR/10))
    # Outdir = "sound/{}dB/dim{}_DNNbased_No{}.wav".format(SNR,dim,index)
    # Outdir = "sound/{}dB/oracle_No{}.wav".format(SNR,index)
    # Outdir = "sound/{}dB/zeli_No{}.wav".format(SNR,index)
    # Outdir = "sound/{}dB/without_No{}.wav".format(SNR,index)
    Outdir = "sound/0dB/allfrq_No2151.wav"
    write(Outdir,fs,NormalizedSignal)

def multinoiseeval():
    dir = "/mnt/aoni02/uchida/multispeaker/area_data/"
    fs, outarea = read(dir+"safia2_No0001.wav")
    fs, areaA = read(dir+"areaA_No0001.wav")
    fs, areaB = read(dir+"areaB_No0001.wav")
    fs, areaC = read(dir+"areaC_No0001.wav")

    outSpec, synp = FFTanalysis.FFTanalysis(outarea)
    ASpec, synp = FFTanalysis.FFTanalysis(areaA)
    BSpec, synp = FFTanalysis.FFTanalysis(areaB)
    CSpec, synp = FFTanalysis.FFTanalysis(areaC)

    # print outSpec.shape, ASpec.shape, BSpec.shape, CSpec.shape
    for i in np.arange(3):
        fs, dist_areaA = read(dir+"areaA_No000{}.wav".format(i+2))
        fs, dist_areaB = read(dir+"areaB_No000{}.wav".format(i+2))
        fs, dist_areaC = read(dir+"areaC_No000{}.wav".format(i+2))
        dist_ASpec, synp = FFTanalysis.FFTanalysis(dist_areaA)
        dist_BSpec, synp = FFTanalysis.FFTanalysis(dist_areaB)
        dist_CSpec, synp = FFTanalysis.FFTanalysis(dist_areaC)
        charlist=["A","B","C"]
        # print dist_ASpec.shape,dist_BSpec.shape,dist_CSpec.shape
        minarg = np.argmin([np.mean(abs(dist_ASpec)**2),np.mean(abs(dist_BSpec)**2),np.mean(abs(dist_CSpec)**2)])
        if minarg == 0:
            minSpec = dist_ASpec
            TrueSpec = ASpec
            selected = "A"
        elif minarg == 1:
            minSpec = dist_BSpec
            TrueSpec = BSpec
            selected = "B"
        elif minarg == 2:
            minSpec = dist_CSpec
            TrueSpec = CSpec
            selected = "C"
        NFRAME, FRQ = np.shape(minSpec)
        plt.subplot(211)
        plt.plot(np.arange(FRQ),abs(minSpec[NFRAME/2]),label="target+noise+dist_{}".format(charlist[i]))
        plt.plot(np.arange(FRQ),abs(TrueSpec[NFRAME/2]),label="target+noise")
        # plt.plot(np.arange(FRQ),abs(outSpec[NFRAME/2]),label="outSpec")
        plt.title("spectrogram NFRAME/2")
        plt.legend()
        plt.subplot(212)
        plt.plot(np.arange(FRQ),abs(np.mean(minSpec,axis=0)),label="target+noise+dist_{}".format(charlist[i]))
        plt.plot(np.arange(FRQ),abs(np.mean(TrueSpec,axis=0)),label="target+noise")
        # plt.plot(np.arange(FRQ),abs(np.mean(outSpec,axis=0)),label="outSpec")
        plt.title("spectrogram meanFRAME")
        plt.legend()
        plt.tight_layout()
        # plt.savefig(dir+"figures/spectrogram.png".format(charlist[i],selected))
        plt.savefig(dir+"figures/dist{}_select{}_spectrogram.png".format(charlist[i],selected))
        plt.clf()

        areaLSD = LSDeval(TrueSpec,minSpec)
        # areaLSD = LSDeval(outSpec,TrueSpec)
        print "dist{}: selected = {}\nareaLSD = {}".format(charlist[i],selected,np.round(areaLSD,decimals = 2))



if __name__ == '__main__':
    # multinoiseeval()
    # m20SNReval()
    SEGANeval()

    SNR = [-20,-10,-5,0,5,10]
    NRR = np.zeros(6)
    LSD = np.zeros(6)
    for i,snr in enumerate(SNR):
        NRR[i],LSD[i] = oneSNReval(snr)
    print np.round(NRR,decimals = 2)
    print np.round(LSD,decimals = 2)

    # SNR = [-20,-10,-5,0,5,10]
    # DIM = [10,20,50,80,100,128]
    # # SNR=[0]
    # meanNRR = np.zeros(6)
    # meanLSD = np.zeros(6)
    # for j,dim in enumerate(DIM):
    #     NRR = np.zeros(6)
    #     LSD = np.zeros(6)
    #     for i,snr in enumerate(SNR):
    #         # DemoSignalNormalize(snr,dim,2151)
    #         NRR[i],LSD[i] = oneSNReval(snr,dim)
    # #     plt1 = plt.figure()
    # #     plt2 = plt.figure()
    # #     plt1.plot(np.zeros(6),NRR,label="dim={}".format(dim))
    # #     plt1.title("NRR")
    # #     plt1.legend()
    # #     plt2.plot(np.zeros(6),LSD,label="dim={}".format(dim))
    # #     plt2.title("LSD")
    # #     plt2.legend()
    #     meanNRR[j] = np.mean(NRR[1:6])
    #     meanLSD[j] = np.mean(LSD[1:6])
    #     print np.round(NRR,decimals = 2)
    #     print np.round(LSD,decimals = 2)
    #     print np.round(meanNRR[j],decimals = 2),np.round(meanLSD[j],decimals = 2)
    # # plt1.savefig("figures/NRR.png")
    # # plt2.savefig("figures/LSD.png")
    #
    # plt.plot(np.zeros(6),meanNRR,label="dim={}".format(dim))
    # plt.title = ("NRR")
    # plt.savefig("figures/NRR.png")
    # plt.clf()
    # plt.plot(np.zeros(6),meanLSD,label="dim={}".format(dim))
    # plt.title = ("LSD")
    # plt.savefig("figures/LSD.png")
