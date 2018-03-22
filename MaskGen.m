%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Generation of Binary Mask for SAFIA
%%% --------------------------------------------------------------------
%%% Mask=safia(ObsSpec1,ObsSpec2,Feature)
%%% --------------------------------------------------------------------
%%% [Input]
%%% ObsSpec1        : Spectrogram of Observed Signal1
%%% ObsSpec2        : Spectrogram of Observed Signal2
%%% Beta            : Flooring parameter (Default : 0)
%%% [Output]
%%% Mask1           : Binary Mask for source 1
%%% Mask2           : Binary Mask for source 2
%%% --------------------------------------------------------------------
%%% * 2013/05/25 Original version
%%% --------------------------------------------------------------------
%%%                             Motoi OMACHI<omachi@pcl.cs.waseda.ac.jp>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Mask1,Mask2]=MaskGen(ObsSpec1,ObsSpec2,Beta,MaskType)

switch nargin
	case 1
		disp('Syntax error');
		return;
	case 2
		Beta=0;
		MaskType='hard';
        case 3
		MaskType='hard';
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

%%% --- Comparison --- %%%
Pow1=ObsSpec1.*conj(ObsSpec1);
Pow1(Pow1<1e-3)=1e-3;
Pow1(isnan(Pow1))=1.0;
Pow1(isinf(Pow1))=1.0;
Pow2=ObsSpec2.*conj(ObsSpec2);
Pow2(Pow2<1e-3)=1e-3;
Pow2(isnan(Pow2))=1.0;
Pow2(isinf(Pow2))=1.0;

%{
Mask1=Pow1./(Pow1+Pow2);
Mask2=Pow2./(Pow1+Pow2);
%}
switch MaskType
     case 'hard'
        Mask1=(Pow1>Pow2);
        Mask2=(Pow2>Pow1);
        Mask1(Mask1==0)=Beta;
        Mask2(Mask2==0)=Beta;
     case 'soft'
        Mask1=Pow1./(Pow1+Pow2);
        Mask2=Pow2./(Pow1+Pow2);
end

return;
