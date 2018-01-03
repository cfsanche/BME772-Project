% SVM CLASSIFIER 

% NOTES
% process feature matrices and split into train and test cases 
% create an svm for each channel and compare each to create a final outcome
% train set should be 150 eeg signals or more for class 1 and class 2
% test the rest and determine classification accuracy

% create final singular output
    % get final svm for the testing case 
    % fuse across all 10 channels - might do 5 to speed things up 
    % apply collar technique (add 0.25s to the beginning and end of each

%load datasets and create train and test sets and class labels
    % nonseizure = 1, seizure = 2
    
%%%%%%% ONE CHANNEL AT a TIME %%%%%%%%    
numfeat = 16;
numsubjects=20; % number of subject
ch=2; % ch = [1 2 9 10 3 4 5 6 11 12 17 18];
nonseizure=cell(1,numsubjects);
seizure=cell(1,numsubjects);
temp1=0; temp2=0;
features = 1:16;%1:numfeat;
%per channel non seizure
for i=1:numsubjects
    
    fname = strcat('sub',num2str(i)); % take features from all subjects for chosen channel 
    
    load(strcat('C:\Users\Camila\Documents\MATLAB\signals\project\data\stuff\ch 3 4\',fname,'_nonseizure.mat')); % load non seizure features that is in workspace 
    
    if (feature_non{ch}==0)
        temp1=i;
        continue; % count how many empty or zero cells there are 
    end
    nonseizure{i}=feature_non{ch}(:,features);

    load(strcat('C:\Users\Camila\Documents\MATLAB\signals\project\data\stuff\ch 3 4\',fname,'_seizure.mat')); %same for seizure 
    
    if (feature_seizure{ch}==0)
        temp2=i; % for example subject 0 had no seizure so this will count for that 
        continue;
    end
    seizure{i}=feature_seizure{ch}(:,features);
    clear feature_non feature_seizure
end

if temp1~=0
    nonseizure(temp1)=[]; % delete empty regions
end
if temp2~=0
    seizure(temp2)=[];
end

num_nonseizure_reg=0; % find total number of num non seizure  
for i=1:length(nonseizure)
        a=length(nonseizure{i});
        num_nonseizure_reg=num_nonseizure_reg+a(1);
        clear a
end
    
num_seizure_reg=0;
for i=1:length(seizure)
    [a,~]=size(seizure{i});
    num_seizure_reg=num_seizure_reg+a;
    clear a
end 

%create dataset

%feature_matrix=zeros(num_nonseizure_reg+num_seizure_reg,numfeat+1); %+1 for class labels
Class1 = zeros(num_nonseizure_reg,numfeat+1); % 1
Class2 = zeros(num_seizure_reg,numfeat+1); % 0
%rng(3);
 train_ind1=randperm(num_nonseizure_reg,400);
 train_ind2=randperm(num_seizure_reg,150);

%train1=zeros(sum(cellfun('length',nonseizure)),1);
start=1;
finish=0;
for i=1:length(nonseizure)
    a=size(nonseizure{i});
    finish=a(1)+finish;
    Class1(start:finish,1:numfeat)=nonseizure{i};
    start=finish+1;
    finish=start-1; 
end
% train1 = cell2mat(nonseizure);                      
Class1(:,end)=1;

 train1=Class1(train_ind1,:);
 test1=Class1;
 test1(train_ind1,:)=[];


start=1;
finish=0;
for i=1:length(seizure)
    a=size(seizure{i});
    finish=a(1)+finish;
    Class2(start:finish,1:numfeat)=seizure{i};
    start=finish+1;
    finish=start-1;
end
Class2(:,end)=ones(num_seizure_reg,1)*2;
train2=Class2(train_ind2,:);
test2=Class2;
test2(train_ind2,:)=[];
r=randperm(length(test2),121);
test2=test2(r,:);

TrainSet = [train1;train2];
TestSet = [test1];

SVMstruct = svmtrain(TrainSet(:,1:numfeat),TrainSet(:,end));

group = svmclassify(SVMstruct,TestSet(:,1:numfeat));

accuracy_svm=length(find(group==TestSet(:,end)))/length(group)*100;

save(['SVMstruct',num2str(10),'.mat'],'SVMstruct');
