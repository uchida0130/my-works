function noising()
  % getteacher();
  % train_noising();
  % test_noising();
  drynoising();
end

function getteacher()
  Fs=16000;
  imp_length = 200;
  peak = 40;
  SPLIT_SIZE = 10000;
  [imp1,fs] = audioread('../anechoic_work_2mic/ch1/imp_data/2mic_ch1_target.wav');
  [imp2,fs] = audioread('../anechoic_work_2mic/ch2/imp_data/2mic_ch2_target.wav');
  if fs~=Fs
    imp1=decimate(imp1,ceil(fs/Fs),512,'fir');
    imp2=decimate(imp2,ceil(fs/Fs),512,'fir');
  end
  [H,I] = max(imp1);
  Indir  = '/mnt/aoni02/uchida/anechoic_work_2mic/origin_data/';
  files_=dir([Indir '*.wav']);

  parfor iter = 1:SPLIT_SIZE,
    [ss,Fs] = audioread([Indir files_(iter).name]);
    out_ = conv(ss,imp1(I-peak:I+imp_length-peak));
    out_ = powerNormalize(out_);
    out = out_(peak:length(out_)-peak);
    audiowrite(['../anechoic_work_2mic/teacher/ch1/No' sprintf('%05d',iter) '.wav'],out,Fs);
  end
  parfor iter = 1:SPLIT_SIZE,
    [ss,Fs] = audioread([Indir files_(iter).name]);
    out_ = conv(ss,imp2(I-peak:I+imp_length-peak));
    out_ = powerNormalize(out_);
    out = out_(peak:length(out_)-peak);
    audiowrite(['../anechoic_work_2mic/teacher/ch2/No' sprintf('%05d',iter) '.wav'],out,Fs);
  end
end

function drynoising()


  Fs=16000;
  noiselen = 16000*(100-5);
  SPLIT_SIZE = 250;
  SNR_list = [-5; 0; 5; 10;];
  noise_list = [9; 11; 13; 14; 18; 20; 26; 28;];
  Indir  = '/mnt/aoni02/uchida/anechoic_work_2mic/origin_data/';
  files_=dir([Indir '*.wav']);
  for SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    disp(['proc: SNR = ' num2str(SNR) '...'])
    parfor noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
        [ss,fs] = audioread([Indir files_((SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter).name]);
        [noise_mono,fs] = audioread(['../JEIDA_NOISE/' sprintf('%02d',NOISE) '.wav']);
        noise = noise_mono(:,1);
        if fs~=Fs
          noise=decimate(noise,ceil(fs/Fs),512,'fir');
        end
        startpoint = randi(noiselen-length(ss))+16000*5;
        noise_ = noise(startpoint:startpoint+length(ss)-1);
        noise_ = Clip(noise_);
        audiowrite(['../anechoic_work_2mic/dry_noisy/train_data/noise_teacher/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noise_,Fs);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/dry_noisy/train_data/noisy/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);

      end
    end
  end

  SNR_list = [-10; -5; 0; 5; 10; 15;];
  noise_list = [30; 47;];
  train_size = 8000;
  SPLIT_SIZE = 100;
  parfor SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    disp(['proc: SNR = ' num2str(SNR) '...'])
    for noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
        [ss,fs] = audioread([Indir files_(train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter).name]);
        [noise_mono,fs] = audioread(['../JEIDA_NOISE/' sprintf('%02d',NOISE) '.wav']);
        noise = noise_mono(:,1);
        if fs~=Fs
          noise=decimate(noise,ceil(fs/Fs),512,'fir');
        end
        startpoint = randi(noiselen-length(ss))+16000*5;
        noise_ = noise(startpoint:startpoint+length(ss)-1);
        noise_ = Clip(noise_);
        audiowrite(['../anechoic_work_2mic/dry_noisy/test_data/noise_teacher/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noise_,Fs);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/dry_noisy/test_data/noisy/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);

      end
    end
  end
end

function train_noising()
  Fs=16000;
  noiselen = 16000*(100-5);
  SPLIT_SIZE = 250;
  SNR_list = [-5; 0; 5; 10;];
  noise_list = [9; 11; 13; 14; 18; 20; 26; 28;];
  ch_num = 2;
  for SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    disp(['proc: SNR = ' num2str(SNR) '...'])
    parfor noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
        [ss,fs] = audioread(['../anechoic_work_2mic/teacher/ch1/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav']);
        [noise,fs] = audioread(['../anechoic_work_2mic/ch1/noise_data/2mic_ch1_noise' sprintf('%02d',NOISE) '.wav']);

        startpoint = randi(noiselen-length(ss))+16000*5;
        noise_ = noise(startpoint:startpoint+length(ss)-1);
        noise_ = Clip(noise_);
        audiowrite(['../anechoic_work_2mic/noise_teacher/train_data/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noise_,Fs);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/ch1/train_data/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);

        [ss,fs] = audioread(['../anechoic_work_2mic/teacher/ch2/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav']);
        [noise,fs] = audioread(['../anechoic_work_2mic/ch2/noise_data/2mic_ch2_noise' sprintf('%02d',NOISE) '.wav']);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/ch2/train_data/No' sprintf('%05d',(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);
      end
    end
  end
end

function test_noising()
  Fs=16000;
  noiselen = 16000*(100-5);
  train_size = 8000;
  SPLIT_SIZE = 100;
  SNR_list = [-10; -5; 0; 5; 10; 15;];
  noise_list = [30; 47;];
  ch_num = 2;
  parfor SNRnum = 1:length(SNR_list),
    SNR = SNR_list(SNRnum);
    disp(['proc: SNR = ' num2str(SNR) '...'])
    for noisenum = 1:length(noise_list),
      NOISE = noise_list(noisenum);
      for iter = 1:SPLIT_SIZE,
        [ss,fs] = audioread(['../anechoic_work_2mic/teacher/ch1/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav']);
        [noise,fs] = audioread(['../anechoic_work_2mic/ch1/noise_data/2mic_ch1_noise' sprintf('%02d',NOISE) '.wav']);

        startpoint = randi(noiselen-length(ss))+16000*5;
        noise_ = noise(startpoint:startpoint+length(ss)-1);
        noise_ = Clip(noise_);
        audiowrite(['../anechoic_work_2mic/noise_teacher/test_data/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noise_,Fs);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/ch1/test_data/' num2str(SNR) 'dB_noise' sprintf('%02d',NOISE) '_No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);

        [ss,fs] = audioread(['../anechoic_work_2mic/teacher/ch2/No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav']);
        [noise,fs] = audioread(['../anechoic_work_2mic/ch2/noise_data/2mic_ch2_noise' sprintf('%02d',NOISE) '.wav']);

        noisy = ss + noise_*sqrt(mean(ss.^2)/mean(noise_.^2)*(10.^(-SNR/10.)));
        noisy = Clip(noisy);
        audiowrite(['../anechoic_work_2mic/ch2/test_data/' num2str(SNR) 'dB_noise' sprintf('%02d',NOISE) '_No' sprintf('%05d',train_size+(SNRnum-1)*length(noise_list)*SPLIT_SIZE+(noisenum-1)*SPLIT_SIZE+iter) '.wav'],noisy,Fs);
      end
    end
  end
end
