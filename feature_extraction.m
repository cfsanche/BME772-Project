tic; % to time how long it takes to run because this does take some time

%only does 1 subject at a time 
for i = 10:20
    name = num2str(i); % Subject
    %sampleRate = 256;  % sampling rate (needed for time conversion)
    load(['EEG_subject0',name]); % Load EEG data
    load(['seizureGT_subject0',name,'.mat']);
    
    % find region where seizure is occuring
    seizure_reg = find(seizureGT==1);
    
    
    % region1 = {'FP1-F7','F7-T7','FP1-F3','F3-C3','T7-FT9'}));
    % region2 = {'FP2-F4','F4-C4','FP2-F8','F8-T8','T8-FT10','FT10-T8'}));
    % region3 = {'T7-P7','P7-O1','C3-P3','P3-O1','P7-T7'}));
    % region4 = {'C4-P4','P4-O2','T8-P8','P8-O2'}));
    % region5 = {'FZ-CZ','CZ-PZ','FT9-FT10'}));
    
    ch = [11 12]; % 2 channels per region chosen at random
    ch_len=length(ch); % numbe of channels
    sig_len= length(EEG(1).ch); % rlength of the signal
    num_feat = 24; % number of features
    
    % this is for the golay filter that i read is good for eeg and set to same
    % values that were used in the paper
    order=3;
    framelen=51;
    
    fs=256; %sampling frequency
    
    %noth filter design taken form matlab tutorial
    d = designfilt('bandstopiir','FilterOrder',2, ...
        'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
        'DesignMethod','butter','SampleRate',fs);
    
    [b,a] = butter(4,0.5/(fs/2),'high'); %remove everything below 0.5Hz for delta
    
    EEGfilt=zeros(sig_len,ch_len); %filtered eeg signal
    k=1;
    for i=ch
        %EEGg=sgolayfilt(EEG(i).ch,order,framelen); %hopefully get rid of noise
        EEGn=filtfilt(d,EEG(i).ch); %notch + golay
        EEGfilt(:,k)=filter(b,a,EEGn); %high pass + notch + golay
        k=k+1;
        clear EEGg EEGn %just in case
    end
    
    EEGseiz = zeros(length(seizure_reg),ch_len); % seizure region
    EEGnon1 = zeros(length(1:min(seizure_reg)-1),ch_len); % region of nonseizure before seizure -> before region
    EEGnon2 = zeros(length(max(seizure_reg)+1:sig_len),ch_len); % after seizure -> after region
    
    % fill them
    for i=1:ch_len
        EEGnon1(:,i)=EEGfilt(1:min(seizure_reg)-1,i);
        EEGnon2(:,i)=EEGfilt(max(seizure_reg)+1:end,i);
        if (isempty(seizure_reg)==0)
            EEGseiz(:,i)=EEGfilt(seizure_reg,i);
        else
            EEGseiz=0;
        end
    end
    
    %% NONSEIZURE REGION
    %floor=rounds to the smallest integer
    region_len=1024; % --> 4s with no overlap
    halfreg_len=region_len/2;
    % make 1024 segments and anything left over will be ignored
    halfs1 = floor(length(EEGnon1(:,1))/region_len); % find number of segments for the before region
    halfs2 = floor(length(EEGnon2(:,1))/region_len); % number of segments in after region
    
    EEGnon1=EEGnon1(1:region_len*halfs1,:);
    EEGnon2=EEGnon2(1:region_len*halfs2,:);
    num_reg=halfs1+halfs2; % total non seizure regions
    
    %will hold the features for non seizure regions for all channels
    feature_non = cell(1,ch_len);
    for i=1:ch_len
        feature_non{i} = zeros(num_reg,num_feat);
    end
    
    %keep track of features
    betamax=1; betamin=2; betamean=3; betastd=4;
    alphamax=5; alphamin=6; alphamean=7; alphastd=8;
    thetamax=9; thetamin=10; thetamean=11; thetastd=12;
    deltamax=13; deltamin=14; deltamean=15; deltastd=16;
    gammamax=17; gammamin=18; gammamean=19; gammastd=20;
    noisemax=21; noisemin=22; noisemean=23; noisestd=24;
    
    for k=1:ch_len
        
        %create even segments from the data for 1 channel at a time
        EEG1=reshape(EEGnon1(:,k),region_len,halfs1);
        EEG2=reshape(EEGnon2(:,k),region_len,halfs2);
        
        for i=1:num_reg %pick segment from either before or after region depending on what iteration we are at
            if i<=halfs1
                EEG_ch_seg = EEG1(:,i);
            else
                EEG_ch_seg = EEG2(:,i-halfs1);
            end
            
            %     %find wavelet coefficients for segments of channel k
            
            [c,l] = wavedec(EEG_ch_seg,5,'db4'); % wavelet decomposition
            [d1,d2,d3,d4,d5] = detcoef(c,l,[1 2 3 4 5]); % detail coefficients
            a5=appcoef(c,l,'db4',5); % approximate coefficients
            
            %fill feature matrix 1 channel at a time for each segment
            feature_non{k}(i,betamax) = max(d3); % 32-16 Hz
            feature_non{k}(i,betamin) = min(d3);
            feature_non{k}(i,betamean) = mean(d3);
            feature_non{k}(i,betastd) = std(d3);
            
            feature_non{k}(i,alphamax) = max(d4); % 16-8 Hz
            feature_non{k}(i,alphamin) = min(d4);
            feature_non{k}(i,alphamean) = mean(d4);
            feature_non{k}(i,alphastd) = std(d4);
            
            feature_non{k}(i,thetamax) = max(d5); % 8-4 Hz
            feature_non{k}(i,thetamin) = min(d5);
            feature_non{k}(i,thetamean) = mean(d5);
            feature_non{k}(i,thetastd) = std(d5);
            
            feature_non{k}(i,deltamax) = max(a5); % 4-0.5 Hz
            feature_non{k}(i,deltamin) = min(a5);
            feature_non{k}(i,deltamean) = mean(a5);
            feature_non{k}(i,deltastd) = std(a5);
            
            feature_non{k}(i,gammamax) = max(d2); % 64-32 Hz
            feature_non{k}(i,gammamin) = min(d2);
            feature_non{k}(i,gammamean) = mean(d2);
            feature_non{k}(i,gammastd) = std(d2);
            
            feature_non{k}(i,noisemax) = max(d1); % 128-64 Hz
            feature_non{k}(i,noisemin) = min(d1);
            feature_non{k}(i,noisemean) = mean(d1);
            feature_non{k}(i,noisestd) = std(d1);
            
            clear d3 d4 d5 a5 EEG_ch_seg c l d2
        end
        clear EEG1 EEG2
    end
    
    %% SEIZURE REGION
    % similar to non seizure
    
    if (EEGseiz~=0) % check if there is a seizure region
        clear num_reg EEG1
        num_reg = floor(length(seizure_reg)/region_len);
        
        EEGs=EEGseiz(1:region_len*num_reg,:);
        feature_seizure = cell(1,ch_len);
        
        for i=1:ch_len
            feature_seizure{i} = zeros(num_reg,num_feat);
        end
        
        for k=1:ch_len
            
            %create segments for channel k
            EEG1=reshape(EEGs(:,k),region_len,num_reg);
            for i=1:num_reg
                EEG_ch_seg = EEG1(:,i);
                
                %     %find wavelet coefficients for segments of channel k
                
                
                [c,l] = wavedec(EEG_ch_seg,5,'db4');
                [d1,d2,d3,d4,d5] = detcoef(c,l,[1 2 3 4 5]);
                
                a5=appcoef(c,l,'db4',5);
                
                feature_seizure{k}(i,betamax) = max(d3); %beta 1
                feature_seizure{k}(i,betamin) = min(d3);
                feature_seizure{k}(i,betamean) = mean(d3);
                feature_seizure{k}(i,betastd) = std(d3);
                
                feature_seizure{k}(i,alphamax) = max(d4);
                feature_seizure{k}(i,alphamin) = min(d4);
                feature_seizure{k}(i,alphamean) = mean(d4);
                feature_seizure{k}(i,alphastd) = std(d4);
                
                feature_seizure{k}(i,thetamax) = max(d5);
                feature_seizure{k}(i,thetamin) = min(d5);
                feature_seizure{k}(i,thetamean) = mean(d5);
                feature_seizure{k}(i,thetastd) = std(d5);
                
                feature_seizure{k}(i,deltamax) = max(a5);
                feature_seizure{k}(i,deltamin) = min(a5);
                feature_seizure{k}(i,deltamean) = mean(a5);
                feature_seizure{k}(i,deltastd) = std(a5);
                
                feature_seizure{k}(i,gammamax) = max(d2);
                feature_seizure{k}(i,gammamin) = min(d2);
                feature_seizure{k}(i,gammamean) = mean(d2);
                feature_seizure{k}(i,gammastd) = std(d2);
                
                feature_seizure{k}(i,noisemax) = max(d1);
                feature_seizure{k}(i,noisemin) = min(d1);
                feature_seizure{k}(i,noisemean) = mean(d1);
                feature_seizure{k}(i,noisestd) = std(d1);
                
                clear d3 d4 d5 a5 EEG_ch_seg c l d2
            end
        end
    else % if not seizure is present return a zero cell array
        feature_seizure = cell(1,ch_len);
        for i=1:ch_len
            feature_seizure{i}=0;
        end
    end
    save(['sub',name,'_nonseizure.mat'],'feature_non');
    save(['sub',name,'_seizure.mat'],'feature_seizure');
    clear all
end
toc;
