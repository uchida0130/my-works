function output = powerNormalize(Input)
  norm = 0.0012;
  [length,ch] = size(Input);
  if ch == 1
    output = Input.*sqrt(norm/mean(Input.^2));
  else
    il = Input(:,1);
    rate = sqrt(norm/mean(il.^2));
    ol = il.*rate;
    ir = Input(:,2);
    or = ir.*rate;
    output = zeros(length,ch);
    output(:,1) = ol;
    output(:,2) = or;
  end
  output = Clip(output);
end
