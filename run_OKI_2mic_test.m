function run_OKI_2mic_test()
  Fs=16000;
  FRAME_SIZE=1024;
  FRAME_SHIFT=256;
  FFTL=1024;
  SPLIT_SIZE=100;
  SNR_list = [-10; -5; 0; 5; 10; 15;];
  noise_list = [30; 47;];
  parfor SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    % disp(['proc: SNR = ' num2str(SNR) '...'])
    for noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
          disp(['proc: SNR = ' num2str(SNR) ', No' sprintf('%04d',8000+(noisenum-1)*SPLIT_SIZE+iter-1) '...'])
          N_mic=2;
          FileDir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch1/test_data/noise' sprintf('%02d',NOISE) '/' num2str(SNR) 'dB/No' sprintf('%04d',8000+(noisenum-1)*SPLIT_SIZE+iter-1) '.wav'];
          signal1 = audioread([FileDir]);
          FileDir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch2/test_data/noise' sprintf('%02d',NOISE) '/' num2str(SNR) 'dB/No' sprintf('%04d',8000+(noisenum-1)*SPLIT_SIZE+iter-1) '.wav'];
          signal2 = audioread([FileDir]);

        %%%%%%%%%%%%%%%%%%%%%%%%
        %%% --- Analysis --- %%%
        %%%%%%%%%%%%%%%%%%%%%%%%


        [X,synparam]=FFTAnalysis([signal1 signal2],FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
        [HFFTL,NFRAME,N_src]=size(X);


        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% --- Beamforming  --- %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Array #1
        Sub1 = (X(:,:,1) - X(:,:,2)).*1;
        Add1 = (X(:,:,1) + X(:,:,2))./2;

        %%%%%%%%%%%%%%%%%%%%%%
        %%% --- Safia  --- %%%
        %%%%%%%%%%%%%%%%%%%%%%
        [Mask11,Mask12]=MaskGen(Add1,Sub1,0,'soft');


        FileDir2 = ['/mnt/aoni02/uchida/anechoic_work_2mic/area_out/test_data/noise' sprintf('%02d',NOISE) '/' num2str(SNR) 'dB/'];
        % Safia output
        % y_out1=Synth(Mask(:,:,1).*X(:,:,1),synparam,300,5500);
        y_out1=Synth(Mask11.*X(:,:,1),synparam,300,5500);
        audiowrite([FileDir2 'No' num2str(8000+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],y_out1 ,16000);

        % y_out3=Synth(Sub1,synparam);
        % audiowrite([FileDir2 'sub_' num2str(SNR) 'dB_noise' sprintf('%02d',noisenum) '_os_test.wav'],y_out3 ,16000);

      end
    end
  end
end
