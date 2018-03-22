function test()
  % inv_length = 55;
  % time_shift = 1;
  % x = [1.,-0.4,0.16,-0.1];
  % h = inv_filter(x,inv_length,time_shift,0);
  % tic
  % y = myfilt(x,h);
  % toc
  % tic
  % y = conv(x,h);
  % toc
  % butterfilt();
  Indir  = '/mnt/aoni02/uchida/anechoic_work_2mic/origin_data/nf001001.wav';
  [ss,fs] = audioread([Indir]);
  norm = mean(ss.^2)


  % signal_cutting();
  % signal_append();
  % stereo_signal();
  % inv_length = 55;
  % time_shift = 1;
  % x = [1.,-0.4,0.16,-0.1];
  % h = inv_filter(x,inv_length,time_shift);
  % y = myfilt(x,h)
end

function h = inv_filter(g,L,m,beta)
  % L = length of inv filter
  % m = time shift of target signal d
  M = length(g);
  G = zeros(M+L-1,L);
  for i = 1:L
    for j = i:M+i-1
      G(j,i) = g(j-i+1);
    end
  end
  d = zeros(M+L-1,1);
  d(m) = 1;
  % h = zeros(L,1);
  h = inv(G'*G+beta.*eye(L))*G'*d;
end

function butterfilt()
  fc = 4000;
  fs = 16000;
  [b,a] = butter(6,fc/(fs/2));
  % freqz(b,a);
  % saveas(gcf,'butter.png')
  % if fs~=Fs
  %         x11=decimate(x11,ceil(fs/Fs),512,'fir');
  d = zeros(2000,1);
  d(20) = 1;
  y = filter(b,a,d);
  plot(1:100,y(1:100));
  saveas(gcf,'butterhakei.png')
end

function [inv_trace, y, error] = calc_inv(inv_length,time_shift,beta)
  Fs=48000;
  FRAME_SIZE=1024;
  FRAME_SHIFT=256;
  FFTL=1024;
  inpulse_length = 1024;
  holding = 20;
  FileDir = ['/mnt/aoni02/uchida/oto/imp/imp_wav/'];
  [G11, fs] = audioread([FileDir 'in_left_out_left.wav']);
  [G12, fs] = audioread([FileDir 'in_left_out_right.wav']);
  [G21, fs] = audioread([FileDir 'in_right_out_left.wav']);
  [G22, fs] = audioread([FileDir 'in_right_out_right.wav']);
  [ma,Index] = max(G11);
  g(:,1)= G11(Index-holding:Index-holding+inpulse_length);
  g(:,2)= G12(Index-holding:Index-holding+inpulse_length);
  g(:,3)= G21(Index-holding:Index-holding+inpulse_length);
  g(:,4)= G22(Index-holding:Index-holding+inpulse_length);
  [G,synparam] = FFTAnalysis(g,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
  % size(mean(G(:,:,1)))
  g11 = G(:,:,1);
  g12 = G(:,:,2);
  g21 = G(:,:,3);
  g22 = G(:,:,4);
  imp = Synth(g11.*g22 - g12.*g21,synparam);
  inv_trace = inv_filter(imp ,inv_length,time_shift,beta);
  y = conv(imp,inv_trace);
  size(y);
  d = zeros(inpulse_length+inv_length,1);
  d(time_shift) = 1;
  error = mean(abs(y-d));
end

function waveform_plot(wave)
  x = 1:length(wave);
  plot(x,wave);
  saveas(gcf,'waveform.png');
end

function [out_signal1,out_signal2] = calc_outsignal(s1,s2,inv_trace)
  Fs=48000;
  FRAME_SIZE=1024;
  FRAME_SHIFT=256;
  FFTL=1024;
  inpulse_length = 1024;
  holding = 20;
  FileDir = ['/mnt/aoni02/uchida/oto/imp/imp_wav/'];
  [G11, fs] = audioread([FileDir 'in_left_out_left.wav']);
  [G12, fs] = audioread([FileDir 'in_left_out_right.wav']);
  [G21, fs] = audioread([FileDir 'in_right_out_left.wav']);
  [G22, fs] = audioread([FileDir 'in_right_out_right.wav']);
  [ma,Index] = max(G11);
  g(:,1)= G11(Index-holding:Index-holding+inpulse_length);
  g(:,2)= G12(Index-holding:Index-holding+inpulse_length);
  g(:,3)= G21(Index-holding:Index-holding+inpulse_length);
  g(:,4)= G22(Index-holding:Index-holding+inpulse_length);
  [G,synparam] = FFTAnalysis(g,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
  [S1,synparam2] = FFTAnalysis(s1,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
  [S2,synparam3] = FFTAnalysis(s2,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
  [INV,synparam4] = FFTAnalysis(inv_trace,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs);
  [a,len1] = size(S1);
  g11 = G(:,:,1);
  g12 = G(:,:,2);
  g21 = G(:,:,3);
  g22 = G(:,:,4);
  [M,I] = max(mean(INV(:,:)));
  inv = INV(:,I);
  out1 = inv*ones(1,len1).*(S1.*(g22*ones(1,len1)) - S2.*(g21*ones(1,len1)));
  out2 = inv*ones(1,len1).*(-S1.*(g12*ones(1,len1)) + S2.*(g11*ones(1,len1)));
  out_signal1 = Synth(out1,synparam2);
  out_signal2 = Synth(out2,synparam3);
end

function stereo_signal()
  FileDir = ['/mnt/aoni02/uchida/oto/'];
  [s1, fs] = audioread([FileDir 'left/left_konnitiwa.wav']);
  [s2, fs] = audioread([FileDir 'right/right_konnitiwa.wav']);
  len=size(s1);
  S(:,1) = s1;
  S(:,2) = s2(1:len(1));
  audiowrite([FileDir '/stereo/konnitiwa_stereo.wav'],S ,48000);
end

function signal_cutting()
  Fs = 48000;
  FileDir = ['/mnt/aoni02/uchida/oto/'];
  [far1, fs] = audioread([FileDir 'left/left_far.wav']);
  [far2, fs] = audioread([FileDir 'right/right_far.wav']);
  [near1, fs] = audioread([FileDir 'left/left_konnitiwa.wav']);
  [near2, fs] = audioread([FileDir 'right/right_konnitiwa.wav']);
  farList = [0; 6; 9; 16; 18; 25; 28; 34; 37; 44; 47; 53; 56; 63; 65; 72;];
  nearList = [0; 6; 7; 12; 13; 19; 20; 25; 27; 32; 33; 39; 40; 45; 47; 52;];
  for i = 1:2:15
    audiowrite([FileDir 'left/left_far_' num2str((i-1)*45/2) '.wav'],far1(farList(i)*fs+1:farList(i+1)*fs) ,fs);
    audiowrite([FileDir 'right/right_far_' num2str((i-1)*45/2) '.wav'],far2(farList(i)*fs+1:farList(i+1)*fs) ,fs);
    audiowrite([FileDir 'left/left_near_' num2str((i-1)*45/2) '.wav'],near1(nearList(i)*fs+1:nearList(i+1)*fs) ,fs);
    audiowrite([FileDir 'right/right_near_' num2str((i-1)*45/2) '.wav'],near2(nearList(i)*fs+1:nearList(i+1)*fs) ,fs);
  end
end

function signal_append()
  Fs = 48000;
  FileDir = ['/mnt/aoni02/uchida/oto/output/time_conv/'];
  BetaList = [0; 0.1; 0.01; 0.02; 0.05; 0.001; 0.002; 0.005];
  for Betanum = 1:8
    beta = BetaList(Betanum);
    [far1, fs] = audioread([FileDir 'left_far_0_conv_beta' num2str(beta) '.wav']);
    [far2, fs] = audioread([FileDir 'right_far_0_conv_beta' num2str(beta) '.wav']);
    [near1, fs] = audioread([FileDir 'left_near_0_conv_beta' num2str(beta) '.wav']);
    [near2, fs] = audioread([FileDir 'right_near_0_conv_beta' num2str(beta) '.wav']);
    z_time = zeros([fs*1 1]);
    for i = 1:7
      [far3, fs] = audioread([FileDir 'left_far_' num2str(i*45) '_conv_beta' num2str(beta) '.wav']);
      [far4, fs] = audioread([FileDir 'right_far_' num2str(i*45) '_conv_beta' num2str(beta) '.wav']);
      [near3, fs] = audioread([FileDir 'left_near_' num2str(i*45) '_conv_beta' num2str(beta) '.wav']);
      [near4, fs] = audioread([FileDir 'right_near_' num2str(i*45) '_conv_beta' num2str(beta) '.wav']);
      A = [far1; z_time; far3;];
      B = [far2; z_time; far4;];
      C = [near1; z_time; near3;];
      D = [near2; z_time; near4;];
      clear far1 far2 near1 near2;
      far1 = A;
      far2 = B;
      near1 = C;
      near2 = D;
      clear A B C D;
    end
    audiowrite([FileDir 'left_far_concat_conv_beta' num2str(beta) '.wav'],far1 ,fs);
    audiowrite([FileDir 'right_far_concat_conv_beta' num2str(beta) '.wav'],far2 ,fs);
    audiowrite([FileDir 'left_near_concat_conv_beta' num2str(beta) '.wav'],near1 ,fs);
    audiowrite([FileDir 'right_near_concat_conv_beta' num2str(beta) '.wav'],near2 ,fs);
  end
end

function y = myfilt(h,x)
  h_rev = flip(h);
  L = length(h_rev);
  n = length(x);
  y = zeros(n+L-1,1);
  H = zeros(n+L-1,n+2*L-2);
  tmp = [h_rev' zeros(1,n+L-2)];
  for i = 1:n+L-1

  end
end

function y = reverse(x)
  n = length(x);
  y = zeros(n,1);
  for i = 1:n
    y(i) = x(n-i+1);
  end
end
