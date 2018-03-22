%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Cardioid Beamformer
%%% -------------------------------------------------------------------- 
%%% OutBF=cardioidBF(ObsSpec1,ObsSpec2,MicInt,Angle,SV,Fs)
%%% -------------------------------------------------------------------- 
%%% [Input]
%%% ObsSpec1        : Spectrogram of Observed Signal1
%%% ObsSpec2        : Spectrogram of Observed Signal2
%%% MicInt          : Mic. interval  [m]      (Default : 0.03)
%%% NullAngle       : Null Angle     [deg]    (Default : 90)
%%% SV              : Sound Velocity [m/s]    (Default : 340)
%%% Fs              : Sampling Frequency [Hz] (Default : 16000)
%%% [Output]
%%% OutBF           : Beamformer output
%%% -------------------------------------------------------------------- 
%%% * 2013/05/25 Original version
%%% --------------------------------------------------------------------
%%%                             Motoi OMACHI<omachi@pcl.cs.waseda.ac.jp>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function OutBF=cardioidBF(ObsSpec1,ObsSpec2,MicInt,NullAngle,SV,Fs)

switch nargin
	case 1
		disp('Syntax error');
		return;
	case 2
		MicInt=0.03;
		NullAngle=pi/2;
		SV=340;
		Fs=16000;
	case 3
		NullAngle=pi/2;
		SV=340;
		Fs=16000;
	case 4
		SV=340;
		Fs=16000;
	case 5
		Fs=16000;
end

%%% --- Exception handling --- %%%
[HFFTL1,NFRAME1]=size(ObsSpec1);
[HFFTL2,NFRAME2]=size(ObsSpec2);
if HFFTL1~=HFFTL2
	disp('Error : FFT Length is different');
	return;
end
if NFRAME1~=NFRAME2
	disp('Error : Number of FRAMES is different');
	return;
end

%%% --- Beamforming --- %%%
FreqVec= ((0:HFFTL1-1)/((HFFTL1-1)*2)*Fs)';
Omega  = 2*pi*FreqVec;
Delay  = MicInt*sin(NullAngle)/SV;

OutBF=ObsSpec1-ObsSpec2.*(exp(-1j*Omega*Delay)*ones(1,NFRAME1));

return;
