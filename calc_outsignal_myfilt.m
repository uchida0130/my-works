%%% [Input]
%%% impFileDir              : 録音したインパルス応答の格納ディレクトリ
%%% s1                      : 目的音声のLEFT側
%%% s2                      : 目的音声のRIGHT側
%%% inv_det                 : 行列式に対する逆フィルタ
%%% [Output]
%%% out_signal1             : 生成したトランスオーラルシステム音声のLEFT側
%%% out_signal2             : 生成したトランスオーラルシステム音声のRIGHT側
%%% myfilt使う方(遅いので注意)

function [out_signal1, out_signal2] = calc_outsignal_myfilt(impFileDir, s1, s2, inv_det)
  Fs=48000;
  % インパルス応答波形の切り出し長
  inpulse_length = 1024;
  % インパルス応答のピークから前20サンプルを切り出しのはじめとする
  holding = 20;
  [G11, fs] = audioread([impFileDir 'in_left_out_left.wav']);
  [G12, fs] = audioread([impFileDir 'in_left_out_right.wav']);
  [G21, fs] = audioread([impFileDir 'in_right_out_left.wav']);
  [G22, fs] = audioread([impFileDir 'in_right_out_right.wav']);
  [ma,Index] = max(G11);
  % インパルス応答波形の切り出し
  g11= G11(Index-holding:Index-holding+inpulse_length);
  g12= G12(Index-holding:Index-holding+inpulse_length);
  g21= G21(Index-holding:Index-holding+inpulse_length);
  g22= G22(Index-holding:Index-holding+inpulse_length);
  out_signal1 = myfilt(inv_det,myfilt(s1,g22)-myfilt(s2,g21));
  out_signal2 = myfilt(inv_det,myfilt(s2,g11)-myfilt(s1,g12));
end
