%%% [Input]
%%% impDir                  : インパルス応答の格納ディレクトリ
%%% inv_length              : 生成する逆フィルタの長さ
%%% delay                   : 教師インパルスの遅延の大きさ
%%% beta                    : 正則化項パラメータ、デフォルトが0.002、推奨は0.05程度まで
%%% LPFon                   : 教師インパルスにローパスフィルターをかけるか選択するオプション、0でオフ
%%% [Output]
%%% inv_det                 : 生成した逆フィルタ
%%% y                       : フィルタ＊逆フィルタの生成波形
%%% error                   : フィルタ＊逆フィルタの生成波形と教師インパルス間の誤差
%%% 実習で制作するmyfilt関数の実行速度に問題があるので、実験内での音声の畳込みにはこちらを用いる

function [inv_det, y, error] = calc_inv_fast(impDir, inv_length, delay, beta, LPFon)
  switch nargin
  case {1,2}
  		disp('Syntax error');
  		return;
  	case 3
  		beta=0.002;
      LPFon = 1;
    case 4
      LPFon = 1;
  end

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
  % インパルスの行列式の計算
  imp_det = conv(g11, g22)-conv(g12, g21);
  % 行列式に対する逆フィルタの計算
  inv_det = inv_filter(imp_det, inv_length, delay, beta, LPFon);
  % フィルタ＊逆フィルタ
  y = conv(imp_det, inv_det);
  % 誤差の計算
  if LPFon == 1
    d = butterfilt((2*impulse_length-1)+inv_length-1,delay);
  else
    d = zeros((2*impulse_length-1)+inv_length-1,1);
    d(delay) = 1;
  end
  error = mean(abs(y-d));
end
