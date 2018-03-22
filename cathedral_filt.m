function cathedral_filt()
  path(path,'ogawa/mfiles');    % 【実験室指定】

  %%% 入力ファイル名(ステレオ)
  is = 'ogawa/ex2_ogawa.wav';

  %%% 出力ファイル名(ステレオ)
  os = 'ogawa/cathedral_ogawa.wav';

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% 【注意】以下は変更しないこと
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %%% 大聖堂のインパルス応答
  [cathedral_impulse,Fs] = audioread(['imp/minster1_000_ortf_48k.wav']);
  % 5秒間分切り出し
  imp_l = cathedral_impulse(1:240000,1);
  imp_r = cathedral_impulse(1:240000,2);

  %%% 収録済み音声をファイルから読み込む(ステレオ)
  disp(['>> Now loading ' is ' ...']);
  % [ss, fs, nbit] = wavread(is);
  [ss, fs] = audioread(is);
  s1 = ss(:,1);
  s2 = ss(:,2);

  S = zeros(length(s1)+240000-1,2);
  % myfilt関数の使用は大変時間がかかるので避ける
  S1 = conv(imp_l,s1);
  S2 = conv(imp_r,s2);
  S(:,1) = powernorm(S1);
  S(:,2) = powernorm(S2);
  % S(:,1) = myfilt(imp_l,voice(:,1));
  % S(:,2) = myfilt(imp_r,voice(:,2));
  audiowrite(os, S, Fs);


end
