function output = powerNormalize(Input)
  Input(Input>=1.0) = 32767/32768;
  Input(Input<-1.0) = -1.0;
  output = Input;
end
