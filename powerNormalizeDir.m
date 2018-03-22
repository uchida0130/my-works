function powerNormalize(Indir)
  norm = 3.5518e-05;
  files_=dir([Indir '*.wav']);
  for i = 1:length(files_)
    fname=files_(i).name;
    [signals,fs] = audioread([Indir fname]);
    signal_out = signals.*sqrt(norm/mean(signals.^2));
    audiowrite([Indir 'matlab_output/' fname],signal_out,fs);
  end
end
