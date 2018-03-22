%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Transformation from Spectrum to Waveform (Overwrap and Add)                                                 
%%% -------------------------------------------------------------------- 
%%% y=Synth(X,SynP,LowFreq,UpFreq)
%%% -------------------------------------------------------------------- 
%%% [Input]
%%% X                       : Spectrogram
%%% SynP                    : Parameters for synthesis
%%% LowFreq                 : LowFreq
%%% UpFreq                  : UpFreq
%%% [Output]
%%% y                       : Waveform
%%% -------------------------------------------------------------------- 
%%% * 2012/02/05 Original version is implemented
%%% --------------------------------------------------------------------
%%%                             Motoi OMACHI<omachi@pcl.cs.waseda.ac.jp>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y=Synth(X,synparm,LowFreq,UpFreq)

if nargin<3
    bpf_=0;
else
    bpf_=1;
    BPF=bpf(synparm.FFTL/2,synparm.Fs,LowFreq,UpFreq);
end

N_samples=synparm.N_samples;
FRAME_SIZE=synparm.FRAME_SIZE;
FRAME_SHIFT=synparm.FRAME_SHIFT;
WINDOW=synparm.WINDOW;
[HFFTL,N_FRAMES]=size(X);
X=[X(1,:);X(2:HFFTL,:);flipud(conj(X(2:HFFTL-1,:)))];

y=zeros(N_samples,1);
for n=1:N_FRAMES
    % --- frame numbers --- %
    bf=1+(n-1)*FRAME_SHIFT;
    ef=bf+FRAME_SIZE-1;
    % --- Overlap and Add --- %
    if bpf_==0
        y_=ifft(X(:,n));
    else
        y_=ifft(X(:,n).*[BPF;flipud(BPF*0)]);
    end
    y_=real(y_(1:FRAME_SIZE));
    y(bf:ef)=y(bf:ef)+y_.*WINDOW;
end
% y=y/1.10371;

return

function fil=bpf(HFFTL,Fs,LowFreq,UpFreq)
    fil=zeros(HFFTL,1);
    N_LowFreq=LowFreq/Fs*(HFFTL*2);
    N_UpFreq=UpFreq/Fs*(HFFTL*2);
    fil(floor(N_LowFreq):floor(N_UpFreq))=1;
    if floor(N_LowFreq)==ceil(N_LowFreq)
        fil(ceil(N_LowFreq))=0.5;
    else
        fil(floor(N_LowFreq))=1/3;
        fil(ceil(N_LowFreq))=2/3;
    end
    if floor(N_UpFreq)==ceil(N_UpFreq)
        fil(ceil(N_UpFreq))=0.5;
    else
        fil(floor(N_UpFreq))=2/3;
        fil(ceil(N_UpFreq))=1/3;
    end
return;
