function output = powernorm(Input)
  M = max(abs(Input));
  output = Input.*(1/M);
end
