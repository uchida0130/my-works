%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FFT analysis                                                 
%%% -------------------------------------------------------------------- 
%%% [X,SynP]=FFTAnalysis(M,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs,WindowType)
%%% -------------------------------------------------------------------- 
%%% [Input]
%%% M                       : Signals
%%% FRAME_SIZE              : Frame size
%%% FRAME_SHIFT             : Frame shift
%%% FFTL                    : FFT length
%%% Fs                      : Sampling Freq.
%%% WindowType              : Window type
%%%                           'hanning' --- hanning window
%%%                           'hamming' --- hamming window
%%%                           'rect' --- rectangular window
%%% [Output]
%%% X                       : Spectrogram
%%% SynP                    : Parameters for synthesis
%%% -------------------------------------------------------------------- 
%%% * 2012/02/05 Original version is implemented
%%% * 2013/04/17 'WindowType' function is implemented
%%% --------------------------------------------------------------------
%%%                             Motoi OMACHI<omachi@pcl.cs.waseda.ac.jp>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X,SynP]=FFTAnalysis(M,FRAME_SIZE,FRAME_SHIFT,FFTL,Fs,WindowType)

switch nargin
    case 1
        FRAME_SIZE=1024; FRAME_SHIFT=256;
        FFTL=1024;       Fs=16000;
        WindowType='hamming';
    case 2
        FRAME_SHIFT=FRAME_SIZE/4;
        FFTL=1024;       Fs=16000;
        WindowType='hamming';
    case 3
        FFTL=1024;       Fs=16000;        
        WindowType='hamming';
    case 4
        Fs=16000;        
        WindowType='hamming';
    case 5
        WindowType='hamming';
end

% --- spectrum analysis --- %
HFFTL=FFTL/2;
[N_samples,N_mics]=size(M);
N_FRAMES=floor((N_samples-FRAME_SIZE)/FRAME_SHIFT)+1;
switch WindowType
    case 'hanning'
        WINDOW=hanning(FRAME_SIZE);
    case 'hamming'
        WINDOW=hamming(FRAME_SIZE);
    case 'rect'
        WINDOW=ones(FRAME_SIZE);
end
X=zeros(HFFTL+1,N_FRAMES,N_mics);
for n=1:N_FRAMES
    % --- frame numbers --- %
    bf=1+(n-1)*FRAME_SHIFT;
    ef=bf+FRAME_SIZE-1;
    % --- Spectrum --- %
    x_n=M(bf:ef,:).*(WINDOW*ones(1,N_mics));
    X_n=fft(x_n,FFTL);
    X_n(HFFTL+2:end,:)=[];
    for mc=1:N_mics
        X(:,n,mc)=X_n(:,mc);
    end
end

SynP.Fs=Fs;
SynP.N_samples=N_samples;
SynP.FRAME_SIZE=FRAME_SIZE;
SynP.FRAME_SHIFT=FRAME_SHIFT;
SynP.N_FRAMES=N_FRAMES;
SynP.FFTL=FFTL;
SynP.WINDOW=WINDOW;
