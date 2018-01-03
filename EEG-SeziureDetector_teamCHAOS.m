% test subject 7
function [seizureMarker_auto] = EEG_SeizureDetector_teamCHAOs(EEG)

ch = [1 2 9 3 5 6 11 12];% only look at these channels
num_feat=1:16;
num_ch=length(ch);
models = cell(1,num_ch);
c=1;
for i=[1 2 3 5 7 8 9 10]
    load(['SVMstruct',num2str(i),'.mat']);
    models{c}=SVMstruct;
    c=c+1;
end
%models = {SVMstruct1,SVMstruct2,SVMstruct3,SVMstruct4,SVMstruct5,SVMstruct6,SVMstruct7,SVMstruct8,SVMstruct9,SVMstruct10};
% make 4 second segments
w = 1024;
sig_len = length(EEG(1).ch);
shift = floor(rem(sig_len,w)/(num_ch-1));
num_reg = floor(sig_len/w);
if shift~=0
    start=1:shift:num_ch*shift;
    finish=num_reg*w:shift:num_reg*w+shift*num_ch;
else
    start=ones(1,num_ch);
    finish=ones(1,num_ch)*sig_len;
end
if length(finish)==11
    finish(11)=[];
end
ch_labels = ones(sig_len,length(ch));

%extract specific channels
EEGsig = zeros(sig_len,length(ch));
for i=1:num_ch
    EEGsig(:,i)=EEG(ch(i)).ch;
end
features=cell(1,num_ch);
clear EEG
fs=256;
d = designfilt('bandstopiir','FilterOrder',2, ...
    'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
    'DesignMethod','butter','SampleRate',fs);

[b,a] = butter(4,0.5/(fs/2),'high'); %remove everything below 0.5Hz for delta

%EEGfilt=zeros(sig_len,num_ch); %filtered eeg signal


for i=1:num_ch
    %EEGg=sgolayfilt(EEGsig(:,i),order,framelen); %hopefully get rid of noise
    EEGn=filtfilt(d,EEGsig(:,i)); %notch + golay
    EEGsig(:,i)=filter(b,a,EEGn); %high pass + notch + golay
    
    clear EEGg EEGn %just in case
end

%create segments outside of main loop for parfor
segmented_signal = cell(1,num_ch);
for k=1:num_ch
    temp=EEGsig(start(k):finish(k),k);
    segmented_signal{k}=reshape(temp,w,num_reg);
    clear temp
end

%keep track of features
betamax=1; betamin=2; betamean=3; betastd=4;
alphamax=5; alphamin=6; alphamean=7; alphastd=8;
thetamax=9; thetamin=10; thetamean=11; thetastd=12;
deltamax=13; deltamin=14; deltamean=15; deltastd=16;
% gammamax=17; gammamin=18; gammamean=19; gammastd=20;
% noisemax=21; noisemin=22; noisemean=23; noisestd=24;

for k=1:num_ch
    
    %create segments for channel k
    for i=1:num_reg
        EEG_ch_seg = segmented_signal{k}(:,i);
        
        %     %find wavelet coefficients for segments of channel k
        
        [c,l] = wavedec(EEG_ch_seg,5,'db4');
        [~,~,d3,d4,d5] = detcoef(c,l,[1 2 3 4 5]);
        
        a5 = appcoef(c,l,'db4',5);
        
        features{k}(i,betamax) = max(d3); %beta 1
        features{k}(i,betamin) = min(d3);
        features{k}(i,betamean) = mean(d3);
        features{k}(i,betastd) = std(d3);
        
        features{k}(i,alphamax) = max(d4);
        features{k}(i,alphamin) = min(d4);
        features{k}(i,alphamean) = mean(d4);
        features{k}(i,alphastd) = std(d4);
        
        features{k}(i,thetamax) = max(d5);
        features{k}(i,thetamin) = min(d5);
        features{k}(i,thetamean) = mean(d5);
        features{k}(i,thetastd) = std(d5);
        
        features{k}(i,deltamax) = max(a5);
        features{k}(i,deltamin) = min(a5);
        features{k}(i,deltamean) = mean(a5);
        features{k}(i,deltastd) = std(a5);
%         
%         features{k}(i,gammamax) = max(d2);
%         features{k}(i,gammamin) = min(d2);
%         features{k}(i,gammamean) = mean(d2);
%         features{k}(i,gammastd) = std(d2);
%         
%         features{k}(i,noisemax) = max(d1);
%         features{k}(i,noisemin) = min(d1);
%         features{k}(i,noisemean) = mean(d1);
%         features{k}(i,noisestd) = std(d1);
        
        clear d3 d4 d5 a5 EEG_ch_seg c l d2 %invalid in parfor
    end
    
end
clear EEGsig

% find class labels
for k=1:num_ch % channel 1
    group = svmclassify(models{k},features{k}(:,num_feat)); %hard code
    for i=1:num_reg
        if i ==1
            s=start(k);
        end
        ch_labels(s:s+w-1,k)=group(i);
        s=s+w;
    end
end

% figure;
% for i=1:num_ch
%     subplot(4,2,i);
%     plot(ch_labels(:,i)); %hold on; plot(seizureGT+1);
% end

final=zeros(1,sig_len);
for i=1:sig_len
    if sum(ch_labels(i,:))==num_ch*2
        final(i)=1;
    else
        final(i)=0;
    end
end
seizureMarker_auto = final;
end
%%% ADD IN VALIDATION FUNCTION %%
% figure;
% plot(final); hold on; plot(seizureGT);
% title('Outcome'); xlabel('Time (s)'); ylabel('Classification');
% legend('Results','Gold Standard');

%accuracy=length(find(final==seizureGT))/length(final)*100;
