%%% [Input]
%%% impFileDir              : 録音したインパルス応答の格納ディレクトリ
%%% inv_length              : 生成する逆フィルタの長さ
%%% time_shift              : 教師インパルスのシフト数
%%% beta                    : 正則化項パラメータ、デフォルトが0.002、推奨は0.05程度まで
%%% LPFon                   : 教師インパルスにローパスフィルターをかけるか選択するオプション、0でオフ
%%% [Output]
%%% inv_det                 : 生成した逆フィルタ
%%% y                       : フィルタ＊逆フィルタの生成波形
%%% error                   : フィルタ＊逆フィルタの生成波形と教師インパルス間の誤差
%%% myfilt使う方(遅いので注意)

function [inv_det, y, error] = calc_inv_myfilt(impFileDir, inv_length, time_shift, beta, LPFon)
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
  % インパルスの行列式の計算
  imp_det = myfilt(g11, g22)-myfilt(g12, g21);
  % 行列式に対する逆フィルタの計算
  inv_det = inv_filter(imp_det, inv_length, time_shift, beta, LPFon);
  % フィルタ＊逆フィルタ
  y = myfilt(imp_det, inv_det);
  % 誤差の計算
  if LPFon == 1
    d = butterfilt(2*inpulse_length+inv_length,time_shift);
  else
    d = zeros(2*inpulse_length+inv_length,1);
    d(time_shift) = 1;
  end
  error = mean(abs(y-d));
end
