function readinmatlab()
  % Indir = '/mnt/aoni02/uchida/work/sound/';
  % Outdir = '/mnt/aoni02/uchida/work/matlab_output/sound/';
  % Indir = '/mnt/aoni02/uchida/anechoic_work_2mic/sound/';
  % Outdir = '/mnt/aoni02/uchida/anechoic_work_2mic/sound/matlab_output/';
  Indir = '/mnt/aoni02/uchida/anechoic_work_2mic/sound/';
  Outdir = '/mnt/aoni02/uchida/anechoic_work_2mic/sound/';
  dBdir = '';
  files_=dir([Indir dBdir '*.wav']);
  parfor i = 1:length(files_)
    fname=files_(i).name;
    [signals,fs] = audioread([Indir dBdir fname]);
    % signals_n = powerNormalize(signals);
    audiowrite([Outdir dBdir fname],signals,fs);
    signals =[];
  end

  % SNRList = [-5; 0; 5; 10;];
  % noiseList = [9; 11; 13; 14; 18; 20; 26; 28;];
  % for SNRnum = 1:4,
  %   SNR = SNRList(SNRnum);
  %   disp(['now proc: ' num2str(SNR) 'dB...'])
  %   parfor noisenum = 1:8,
  %     NOISE = noiseList(noisenum);
  %     Indir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch1/' num2str(SNR) 'dB_mixed_data/noise' sprintf('%02d',NOISE) '/'];
  %     Outdir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch1/' num2str(SNR) 'dB_mixed_data/noise' sprintf('%02d',NOISE) '/'];
  %     dBdir = '';
  %     files_=dir([Indir dBdir '*.wav']);
  %     for i = 1:length(files_)
  %       fname=files_(i).name;
  %       [signals,fs] = audioread([Indir dBdir fname]);
  %       % signals_n = powerNormalize(signals);
  %       audiowrite([Outdir dBdir fname],signals,fs);
  %       signals = [];
  %     end
  %   end
  % end
  %
  % SNRList = [-10; -5; 0; 5; 10; 15;];
  % noiseList = [30; 47;];
  % parfor SNRnum = 1:6,
  %   SNR = SNRList(SNRnum);
  %   for noisenum = 1:2,
  %     NOISE = noiseList(noisenum);
  %     Indir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch1/test_data/noise' sprintf('%02d',NOISE) '/' num2str(SNR) 'dB/'];
  %     Outdir = ['/mnt/aoni02/uchida/anechoic_work_2mic/ch1/test_data/noise' sprintf('%02d',NOISE) '/' num2str(SNR) 'dB/'];
  %     dBdir = '';
  %     files_=dir([Indir dBdir '*.wav']);
  %     for i = 1:length(files_)
  %       fname=files_(i).name;
  %       [signals,fs] = audioread([Indir dBdir fname]);
  %       % signals_n = powerNormalize(signals);
  %       audiowrite([Outdir dBdir fname],signals,fs);
  %       signals = [];
  %     end
  %   end
  % end

  % SNRList = [-10; -5; 0; 5; 10; 15;];
  % noiseList = [9; 11; 13; 14; 18; 20; 26; 28; 30; 47;];
  % for SNRnum = 1:length(SNRList),
  %   SNR = SNRList(SNRnum);
  %   disp(['now proc: ' num2str(SNR) 'dB...'])
  %   parfor noisenum = 1:length(noiseList),
  %     NOISE = noiseList(noisenum);
  %     Indir = ['/mnt/aoni02/uchida/anechoic_work_2mic/noise_teacher/' num2str(SNR) 'dB/noise' sprintf('%02d',NOISE) '/'];
  %     Outdir = ['/mnt/aoni02/uchida/anechoic_work_2mic/noise_teacher/' num2str(SNR) 'dB/noise' sprintf('%02d',NOISE) '/'];
  %     dBdir = '';
  %     files_=dir([Indir dBdir '*.wav']);
  %     for i = 1:length(files_)
  %       fname=files_(i).name;
  %       [signals,fs] = audioread([Indir dBdir fname]);
  %       % signals_n = powerNormalize(signals);
  %       audiowrite([Outdir dBdir fname],signals,fs);
  %       signals =[];
  %     end
  %   end
  % end
  % powerNormalize(Indir);
end
