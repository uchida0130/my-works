%%% [Input]
%%% h                       : インパルス応答の波形
%%% x                       : 畳み込みたい目的音の波形
%%% [Output]
%%% y                       : 畳み込みした波形
% conv()と等価　処理時間がダメなやつ
% みんなが実習で作るやつ


function y = myfilt(h,x)
  h_rev = flip(h);
  L = length(h_rev);
  n = length(x);
  y = zeros(n+L-1,1);
  tmp = [zeros(L-1,1);x;zeros(L-1,1)];
  for i = 1:n+L-1
    y(i) = tmp(i:i+L-1)'*h_rev;
  end
end
