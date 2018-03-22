%%% [Input]
%%% imp                     : 逆フィルタを求めたいインパルス応答
%%% inv_length              : 生成する逆フィルタの長さ
%%% delay                   : 教師インパルスの遅延の大きさ
%%% beta                    : 正則化項パラメータ、デフォルトが0.002、推奨は0.05程度まで
%%% LPFon                   : 教師インパルスにローパスフィルターをかけるか選択するオプション、0でオフ
%%% [Output]
%%% h                       : 生成した逆フィルタ

function h = inv_filter(imp, inv_length, delay, beta, LPFon)
  M = length(imp);
  % インパルス応答をずらしてならべた行列Gの生成
  G = zeros(M+inv_length-1, inv_length);
  for i = 1:inv_length
    for j = i:M+i-1
      G(j, i) = imp(j-i+1);
    end
  end
  if LPFon == 1
    d = butterfilt(M+inv_length-1, delay);
  else
    d = zeros(M+inv_length-1, 1);
    d(delay) = 1;
  end
  % 正則化項を含めた逆フィルタの計算
  h = inv(G'*G+beta.*eye(inv_length))*G'*d;
end
