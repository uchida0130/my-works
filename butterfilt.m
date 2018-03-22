%%% [Input]
%%% d_length                : 生成するインパルス波形の長さ
%%% delay                   : インパルスの遅延
%%% [Output]
%%% y                       : 生成した6次バタワースLPFのインパルス波形

function y = butterfilt(d_length,delay)
  % カットオフ周波数
  fc = 8000;
  % サンプリング周波数
  fs = 48000;
  % バタワースフィルタのフィルタ係数
  [b,a] = butter(6,fc/(fs/2));
  % バタワースフィルタの図示(確認用)
  % freqz(b,a);
  % saveas(gcf,'butter.png')
  d = zeros(d_length,1);
  d(delay) = 1;
  y = filter(b,a,d);
  % 生成LPFインパルス波形の図示(確認用)
  % plot(delay-10:delay+100,y(delay-10:delay+100));
  % saveas(gcf,'butterhakei.png')
end
