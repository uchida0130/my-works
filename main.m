function main(name)

  filenameList = [cellstr('kaiten'); cellstr('a--'); ];

  if nargin ~= 0
    filenameList = [cellstr(name);];
  end
  % 正則化パラメータのリスト
  BetaList = [0.002; 0.005; 0.02;];
  Fs=48000;
  for filenum = 1:length(filenameList)
    filename = char(filenameList(filenum));
    disp(['proc: ' filename '...'])
    for Betanum = 1:length(BetaList)
      beta = BetaList(Betanum);
      % 逆フィルタの長さ
      inv_length = 6000;
      % 教師インパルスの遅延
      delay = 900;
      %　インパルス応答の格納ディレクトリ
      impDir = ['imp/imp_wav_cut/'];

      %%%%%%%%%%%%%%%%%%%%%%%%
      %%%%% 逆フィルタの生成 %%%%%
      %%%%%%%%%%%%%%%%%%%%%%%%
      % 出力は逆フィルタ、フィルタと逆フィルタの畳み込み、その誤差
      [inv_det, y, error] = calc_inv_fast(impDir, inv_length, delay, beta);
      % フィルタ＊逆フィルタの生成波形、誤差の出力(確認用)
      % y(delay-20:delay+80);
      % max(y)
      % error

      %%%%%%%%%%%%%%%%%%%%%%%%
      %%% システム目的音の生成 %%%
      %%%%%%%%%%%%%%%%%%%%%%%%
      [s1, fs] = audioread(['left/left_' filename '.wav']);
      [s2, fs] = audioread(['right/right_' filename '.wav']);
      % [s1, fs, nbit] = wavread(['left/left_' filename '.wav']);
      % [s2, fs, nbit] = wavread(['right/right_' filename '.wav']);
      [x1, x2] = calc_outsignal_fast(impDir, s1, s2, inv_det);
      x1_n = powerNormalize(x1);
      if exist('output') ~= 7
        mkdir('output');
      end
      audiowrite(['output/left_' filename '_beta' num2str(beta) '.wav'], x1_n, Fs);
      % wavwrite( x1_n, Fs, ['output/left_' filename '_beta' num2str(beta) '.wav']);
      x2_n = powerNormalize(x2);
      audiowrite(['output/right_' filename '_beta' num2str(beta) '.wav'], x2_n, Fs);
      % wavwrite( x2_n, Fs, ['output/right_' filename '_beta' num2str(beta) '.wav']);
      len=length(x1_n);
      S = zeros(len,2);
      S(:,1) = x1_n;
      S(:,2) = x2_n;
      audiowrite(['output/stereo_' filename '_beta' num2str(beta) '.wav'], S, Fs);
      % wavwrite( S, Fs, ['output/stereo_' filename '_beta' num2str(beta) '.wav']);
    end
  end
end
