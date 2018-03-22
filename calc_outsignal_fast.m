%%% [Input]
%%% impDir                  : インパルス応答の格納ディレクトリ
%%% s1                      : 目的音声のLEFT側
%%% s2                      : 目的音声のRIGHT側
%%% inv_det                 : 行列式に対する逆フィルタ
%%% [Output]
%%% out_signal1             : 生成したトランスオーラルシステム音声のLEFT側
%%% out_signal2             : 生成したトランスオーラルシステム音声のRIGHT側
%%% 実習で制作するmyfilt関数の実行速度に問題があるので、実験内での音声の畳込みにはこちらを用いる

function [out_signal1, out_signal2] = calc_outsignal_fast(impDir, s1, s2, inv_det)
  Fs=48000;
  [g11, fs] = audioread([impDir 'in_left_out_left.wav']);
  [g12, fs] = audioread([impDir 'in_left_out_right.wav']);
  [g21, fs] = audioread([impDir 'in_right_out_left.wav']);
  [g22, fs] = audioread([impDir 'in_right_out_right.wav']);
  % [g11, fs, nbit] = wavread([impDir 'in_left_out_left.wav']);
  % [g12, fs, nbit] = wavread([impDir 'in_left_out_right.wav']);
  % [g21, fs, nbit] = wavread([impDir 'in_right_out_left.wav']);
  % [g22, fs, nbit] = wavread([impDir 'in_right_out_right.wav']);
  impulse_length = length(g11);
  inv_length = 6000;
  delay = 900;
  d = butterfilt((2*impulse_length-1)+inv_length-1,delay);
  LP_s1 = conv(s1,d);
  LP_s2 = conv(s2,d);
  out_signal1 = conv(inv_det,conv(LP_s1,g22)-conv(LP_s2,g21));
  out_signal2 = conv(inv_det,conv(LP_s2,g11)-conv(LP_s1,g12));
end
