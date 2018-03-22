function run_OKI_2mic()
  train_AE();
  test_AE();
end

function train_AE()
  Fs=16000;
  FRAME_SIZE=1024;
  FRAME_SHIFT=256;
  FFTL=1024;
  SPLIT_SIZE=250;
  SNR_list = [-5; 0; 5; 10;];
  noise_list = [9; 11; 13; 14; 18; 20; 26; 28;];
  mask_type = 'soft';
  for SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    disp(['proc: SNR = ' num2str(SNR) '...'])
    parfor noisenum = 1:length(noise_list),
      for iter = 1:SPLIT_SIZE,
          % disp(['proc: SNR = ' num2str(SNR) ', No' sprintf('%04d',(noisenum-1)*SPLIT_SIZE+iter-1) '...'])
          N_mic=2;

          FileDir = ['../anechoic_work_2mic/ch1/train_data/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          signal1 = audioread([FileDir]);
          FileDir = ['../anechoic_work_2mic/ch2/train_data/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          signal2 = audioread([FileDir]);

          %%%%%%%%%%%%%%%%%%%%%%%%
          %%% --- Analysis --- %%%
          %%%%%%%%%%%%%%%%%%%%%%%%
          %%% --- FFT --- %%%
          % Array #1 (1-1,1-2)
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
          [Mask11,Mask12]=MaskGen(Add1,Sub1,0,mask_type);

          teachDir = ['../anechoic_work_2mic/teacher/ch1/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          teacher = audioread([teachDir]);
          [T,synpT]=FFTAnalysis(teacher,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
          teacher_ = Synth(Mask11.*T,synpT,300,5500);
          Pnorm= sqrt(mean(teacher.^2)/mean(teacher_.^2));

          if mask_type == 'hard'
            FileDir2 = ['/mnt/aoni02/uchida/anechoic_work_2mic/hard_area_out/train_data/'];
          else
            FileDir2 = ['/mnt/aoni02/uchida/anechoic_work_2mic/area_out/train_data/'];
          end
          % Safia output
          % y_out1=Synth(Mask(:,:,1).*X(:,:,1),synparam,300,5500);
          y_out1 = Synth(Mask11.*X(:,:,1),synparam,300,5500).*Pnorm;
          y_out1 = Clip(y_out1);
          audiowrite([FileDir2 'No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],y_out1 ,16000);
          % y_out2=Synth(Mask(:,:,2).*X(:,:,1),synparam,300,5500);
          % y_out2=Synth(Mask12.*X(:,:,1),synparam,300,5500);
          % audiowrite([FileDir2 'noisy_No' sprintf('%04d',iter-1) '.wav'],y_out2 ,16000);

      end
    end
  end
end

function test_AE()
  Fs=16000;
  train_size = 8000;
  FRAME_SIZE=1024;
  FRAME_SHIFT=256;
  FFTL=1024;
  SPLIT_SIZE=100;
  SNR_list = [-10; -5; 0; 5; 10; 15;];
  noise_list = [30; 47;];
  mask_type = 'soft';
  parfor SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    for noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
          % disp(['proc: SNR = ' num2str(SNR) ', No' sprintf('%04d',(noisenum-1)*SPLIT_SIZE+iter-1) '...'])
          N_mic=2;
          FileDir = ['../anechoic_work_2mic/ch1/test_data/' num2str(SNR) 'dB_noise' sprintf('%02d',NOISE) '_No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          signal1 = audioread([FileDir]);
          FileDir = ['../anechoic_work_2mic/ch2/test_data/' num2str(SNR) 'dB_noise' sprintf('%02d',NOISE) '_No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          signal2 = audioread([FileDir]);

          %%%%%%%%%%%%%%%%%%%%%%%%
          %%% --- Analysis --- %%%
          %%%%%%%%%%%%%%%%%%%%%%%%
          %%% --- FFT --- %%%
          % Array #1 (1-1,1-2)
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
          [Mask11,Mask12]=MaskGen(Add1,Sub1,0,mask_type);

          teachDir = ['../anechoic_work_2mic/teacher/ch1/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'];
          teacher = audioread([teachDir]);
          [T,synpT]=FFTAnalysis(teacher,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
          teacher_ = Synth(Mask11.*T,synpT,300,5500);
          Pnorm= sqrt(mean(teacher.^2)/mean(teacher_.^2));

          if mask_type == 'hard'
            FileDir2 = ['/mnt/aoni02/uchida/anechoic_work_2mic/hard_area_out/test_data/'];
          else
            FileDir2 = ['/mnt/aoni02/uchida/anechoic_work_2mic/area_out/test_data/'];
          end
          % Safia output
          % y_out1=Synth(Mask(:,:,1).*X(:,:,1),synparam,300,5500);
          y_out1=Synth(Mask11.*X(:,:,1),synparam,300,5500).*Pnorm;
          y_out1 = Clip(y_out1);
          audiowrite([FileDir2  num2str(SNR) 'dB_noise' sprintf('%02d',NOISE) '_No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],y_out1 ,16000);
          % y_out2=Synth(Mask(:,:,2).*X(:,:,1),synparam,300,5500);
          % y_out2=Synth(Mask12.*X(:,:,1),synparam,300,5500);
          % audiowrite([FileDir2 'noisy_No' sprintf('%04d',iter-1) '.wav'],y_out2 ,16000);

      end
    end
  end
end
