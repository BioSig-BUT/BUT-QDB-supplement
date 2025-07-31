% TEST RUNNER                                                        7/2025
% -------------------------------------------------------------------------
% for Brno University of Technology ECG Quality Database (BUT QDB)
% Data available at: https://physionet.org/content/butqdb/1.0.0/
% Code available at: https://github.com/BioSig-BUT/BUT-QDB-supplement
%                                                        
% The script is intended for signals and annotation reading, evaluating the 
% performance of algorithms designed for estimating the quality of ECG signals 
% on BUT QDB. The script can be divided into four parts in terms of functionality:
%
% 1) Data Reading
% The script provides sequential reading of the annotated parts of ECG
% signals including annotations. For loading the ECG signals, the script uses 
% the "rdsamp" function, which is part of the WFDB Toolbox and must be installed 
% in Matlab. Instructions can be found here: 
% https://archive.physionet.org/physiotools/matlab/wfdb-app-matlab/. To read 
% the corresponding annotations, the ann_reader_v02.m function is used, which is 
% integral part of this script. We designed a pre-division into a training 
% part and a testing part in a ratio of about 65% to 35%. To load all/train/test 
% data "choseDATA" function is used and as an input you select ‘all’/’train’/’test’ 
% in the “Settings” section of this code.
%
% 2) Tested ECG quality estimation function
% Put the custom algorithm for ECG quality estimation in section "Quality estimation 
% algorithm". The mandatory output is a column vector 'q' of the same length as the ECG
% signal, containing quality classes 1, 2, or 3. The inputs of the function are at the 
% user's choice. As an example, there is a function in the script that generates quality 
% classes randomly.
%
% 3) Waveform display option
% In the "Settings" section, you can select the option to display waveforms 
% of the individual parts of the ECG signal along with the 3-axis ACC signal and 
% quality annotations provided by the experts and also by your proposed algorithm. 
%
% 4) Results
% The results are divided into two parts, the statistical results and the results 
% of the analysis of the quality segments newly created by the tested algorithm. The 
% statistical results calculated by the script include confusion matrix, F1 score for 
% each quality class, overall accuracy, errors of type I and II, and processing time. 
% The script also provides a more detailed breakdown of results for each individual 
% record. For each record as well as for the whole selected dataset, it displays the 
% duration, the distribution of Q1/Q2/Q3 segments based on annotations, F1 scores for 
% Q1/Q2/ Q3, as well as overall accuracy.
% Segmentation analysis examines the number of segments created, the maximum, 
% minimum, mean and median segment duration. These statistics are then compared 
% between the actual quality annotations provided by the experts and predictions 
% according to the tested algorithm.
% All results are displayed in the Matlab command line and are also saved in a 
% separate *.xlsx file. The results of each run are stored in a separate sheet 
% labeled with date and time. 
%
% Created by authors of BUT QDB, 7/2025
% MIT License
% -------------------------------------------------------------------------

close all; clc; clear
% -- Settings -------------------------------------------------------------
plot_waveforms = 0;         % Select (0/1) if the user wants to plot ECG waveforms and quality annotations

% Select whether you want to test your algorithm on all/train/test data by 
% inserting appropriate parameter in "choseDATA" function (Primarily intended for test data)
REC = choseDATA('test');    % all / train / test
% -------------------------------------------------------------------------

L = 0;
for i = 1 : size(REC,1)
    for j = 2:2:length(REC(i,~isnan(REC(i,:))))
        L = L + REC(i,j+1) - REC(i,j) + 1;
    end
end

ANN = cell(size(REC,1),1);  Q = ANN; Ln = 0;
w = waitbar(0,'Processing...'); tic;
for i = 1 : size(REC,1)
    part = 0;
    for j = 2:2:length(REC(i,~isnan(REC(i,:))))

        ID = num2str(REC(i,1));
        part = part+1;
     	from = REC(i,j); from_acc = round(REC(i,j)/10)+1;
      	to = REC(i,j+1); to_acc = round(REC(i,j+1)/10);
        
        [ecg,fs_ecg,tm_ecg]=rdsamp([ID,'\',ID,'_ECG'],[],to,from,0);   % rdsamp is function from WFDB toolbox

        ann  = ann_reader_v02([ID,'\',ID,'_ANN'],4,from,to);    % anntr = 4; 4 is consensus of anotators
     	ann = [int8(ann); 4]; 
        ANN{i,1} = [ANN{i,1};ann];
              
    % --- Quality estimation algorithm ------------------------------------
        % The output is expected to be a column vector 'q' of the same length 
        % as the ECG signal. The 'q' vector has ONLY the values 1,2 or 3, which 
        % reflect the quality classes Q1, Q2 and Q3, respectively.
        
        q = randi([1 3],1,length(ecg))';
    % ---------------------------------------------------------------------
        q = [int8(q); 4]; 
        Q{i,1} = [Q{i,1};q];
        
        if plot_waveforms
            figure;
            [acc,fs_acc,tm_acc]=rdsamp([ID,'\',ID,'_ACC'],[],to_acc,from_acc,0);
            ax1 = subplot(3,1,1); plot(tm_ecg,ecg); xlabel('Time [s]'); ylabel('Voltage [\muV]')
            title(sprintf('Record %0.0f, part %0.0f.',REC(i,1),part)); legend('ECG')
            ax2 = subplot(3,1,2); plot(tm_acc,acc(:,1)); hold on; plot(tm_acc,acc(:,2)); plot(tm_acc,acc(:,3));
            xlabel('Time [s]'); ylabel('ACC [mili-g]'); legend('X-axis','Y-axis','Z-axis')
            ax3 = subplot(3,1,3); plot(tm_ecg,ann(1:end-1)); hold on; plot(tm_ecg,q(1:end-1)); 
            xlabel('Time [s]'); ylabel('Quality class [-]'); legend('Actual','Predicted')
            linkaxes([ax1 ax2 ax3],'x'); ylim([0.5 3.5]); 
        end
        Ln = Ln + to - from + 1; 
        waitbar(Ln/L,w,sprintf('%0.0f: Record %0.0f-%0.0f, Time: %0.0f sec (%0.0f %%)',i,REC(i,1),part,toc,Ln/L*100));
    end
end
proc_time = toc;
waitbar(Ln/L,w,sprintf('%0.0f %%, Time: %0.0f sec',Ln/L*100,proc_time));

% -------------------------- RESULTS --------------------------------------
fileName = 'TESTresults.xlsx';
Sheet = strrep(string(datetime('now')),':','-');
txt1 = {'All together','','','','','Individually by signals','','','','','','','','','','Segmentation analysis'};
txt2 = {'','Confusion matrix [%]','','','','','','Distribution [%]','','','F1 [%]','','','ACC [%]','','','Actual','Predicted'};
txt3 = {'','Pred Q1','Pred Q2','Pred Q3','','ID','Dur [min]','Q1','Q2','Q3','Q1','Q2','Q3'};
txt4 = {'Act Q1','Act Q2','Act Q3','','F1 Q1','F1 Q2','F1 Q3','','Proc time'}'; txt5 = {'ACC','Err I type','Err II type'}';
txt6 = {'# segs','Max Dur','Min Dur','Mea Dur','Med Dur'}'; txt7 = {'[-]','[sec]','[sec]','[sec]','[sec]'}';

writecell(txt1,fileName,'Sheet',Sheet,'Range','A1','AutoFitWidth',0); writecell(txt2,fileName,'Sheet',Sheet,'Range','A3','AutoFitWidth',0)
writecell(txt3,fileName,'Sheet',Sheet,'Range','A4','AutoFitWidth',0); writecell(txt4,fileName,'Sheet',Sheet,'Range','A5','AutoFitWidth',0)
writecell(txt5,fileName,'Sheet',Sheet,'Range','C9','AutoFitWidth',0); writecell(txt6,fileName,'Sheet',Sheet,'Range','P5','AutoFitWidth',0)
writecell(txt7,fileName,'Sheet',Sheet,'Range','S5','AutoFitWidth',0); writecell({'[min]'},fileName,'Sheet',Sheet,'Range','C13','AutoFitWidth',0)
writecell({'ALL'},fileName,'Sheet',Sheet,'Range',['F', num2str(5+size(REC,1))],'AutoFitWidth',0);

% -- STATISTICAL ANALYSIS -------------------------------------------------
[ConfM, ACC, E, F1, ~, ~] = summarySTAT(cat(1,ANN{:}),cat(1,Q{:}),fs_ecg);

fprintf('STATISTICAL ANALYSIS\n')
fprintf('Confusion matrix [%%]\n')
fprintf('     Q1     Q2     Q3\n')
fprintf('Q1  %05.2f  %05.2f  %05.2f \n',ConfM(1,:))
fprintf('Q2  %05.2f  %05.2f  %05.2f \n',ConfM(2,:))
fprintf('Q3  %05.2f  %05.2f  %05.2f \n\n',ConfM(3,:))
fprintf('ACC =         %05.2f %% \n',ACC)
fprintf('Err I type =  %05.2f %% \n',E(1))
fprintf('Err II type = %05.2f %% \n',E(2))
fprintf('F1 Q1 =       %05.2f %% \n',F1(1))
fprintf('F1 Q2 =       %05.2f %% \n',F1(2))
fprintf('F1 Q3 =       %05.2f %% \n',F1(3))
fprintf('Proc time =   %0.0fmin %0.0fsec \n\n', floor(proc_time/60),rem(proc_time,60));

writecell(num2cell(ConfM),fileName,'Sheet',Sheet,'Range','B5','AutoFitWidth',0);
writecell(num2cell(F1),fileName,'Sheet',Sheet,'Range','B9','AutoFitWidth',0);
writecell(num2cell([ACC;E(1);E(2)]),fileName,'Sheet',Sheet,'Range','D9','AutoFitWidth',0);
writecell(num2cell(proc_time/60),fileName,'Sheet',Sheet,'Range','B13','AutoFitWidth',0);

fprintf('Individually by signals\n')
fprintf('ID       Dur      Distrib Q1/Q2/Q3      F1 score Q1,Q2,Q3      ACC\n')
for i = 1:size(REC,1)
    [~, ACC, ~, F1, tim, Q123] = summarySTAT(ANN{i,1},Q{i,1},fs_ecg);
    fprintf('%0.0f   %0.0f min   %05.2f/%05.2f/%05.2f %%   %05.2f,%05.2f,%05.2f %%   %05.2f %% \n',REC(i,1),tim,Q123,F1,ACC)
    writecell(num2cell([REC(i,1) tim Q123(1) Q123(2) Q123(3) F1(1) F1(2) F1(3) ACC]), fileName, 'Sheet', Sheet,'Range', ['F', num2str(4+i)],'AutoFitWidth',0);
end
[ConfM, ACC, E, F1, tim, Q123] = summarySTAT(cat(1,ANN{:}),cat(1,Q{:}),fs_ecg);
fprintf('ALL      %0.0f min   %05.2f/%05.2f/%05.2f %%   %05.2f,%05.2f,%05.2f %%   %05.2f %% \n',tim,Q123,F1,ACC)
writecell(num2cell([tim Q123(1) Q123(2) Q123(3) F1(1) F1(2) F1(3) ACC]), fileName, 'Sheet', Sheet,'Range', ['G', num2str(5+i)],'AutoFitWidth',0);

% -- SEGMENTATION ANALYSIS ------------------------------------------------
SegAna = summarySEGS(cat(1,ANN{:}),cat(1,Q{:}),fs_ecg);

fprintf('\nSEGMENTATION ANALYSIS \n')
fprintf('             Actual      Predicted \n')
fprintf('# segs      %0.0f        %0.0f        [-] \n',SegAna(1,:))
fprintf('Max Dur     %0.2f     %0.2f     [sec] \n',SegAna(2,:))
fprintf('Min Dur     %0.2f        %0.2f        [sec] \n',SegAna(3,:))
fprintf('Mean Dur    %0.2f       %0.2f       [sec] \n',SegAna(4,:))
fprintf('Median Dur  %0.2f        %0.2f        [sec] \n',SegAna(5,:))
writecell(num2cell(SegAna), fileName, 'Sheet', Sheet,'Range', 'Q5','AutoFitWidth',0);

% -- ADDITIONAL FUNCTIONS -------------------------------------------------
function REC = choseDATA(part)
switch part
    case 'all'
        REC(1,:) =  [100001,1       ,87087000,NaN(1,6)];
        REC(2,:) =  [100002,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(3,:) =  [103001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(4,:) =  [103002,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(5,:) =  [103003,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(6,:) =  [104001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(7,:) =  [105001,46800001,139147000,NaN(1,6)];
        REC(8,:) =  [111001,1       ,90645000,NaN(1,6)];
        REC(9,:) =  [113001,28800001,30000000,36120000,37319999,57600001,58800000,NaN(1,2)];
        REC(10,:) = [114001,11214751,11334751,11674751,12874750,28800001,30000000,57600001,58800000];
        REC(11,:) = [115001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(12,:) = [118001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(13,:) = [121001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(14,:) = [122001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(15,:) = [123001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(16,:) = [124001,28800001,30000000,33700000,34899999,57600001,58800000,65100000,66299999];
        REC(17,:) = [125001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(18,:) = [126001,28800001,30000000,57600001,58800000,NaN(1,4)];
    case 'train'
        REC(1,:) = [100001,1       ,87087000,NaN(1,6)];
        REC(2,:) = [100002,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(3,:) = [103001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(4,:) = [103002,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(5,:) = [103003,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(6,:) = [104001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(7,:) = [111001,1       ,90645000,NaN(1,6)];
        REC(8,:) = [113001,28800001,30000000,36120000,37319999,57600001,58800000,NaN(1,2)];
        REC(9,:) = [114001,11214751,11334751,11674751,12874750,28800001,30000000,57600001,58800000];
        REC(10,:) = [115001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(11,:) = [126001,28800001,30000000,57600001,58800000,NaN(1,4)];
    case 'test'
        REC(1,:) = [105001,46800001,139147000,NaN(1,6)];
        REC(2,:) = [118001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(3,:) = [121001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(4,:) = [122001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(5,:) = [123001,28800001,30000000,57600001,58800000,NaN(1,4)];
        REC(6,:) = [124001,28800001,30000000,33700000,34899999,57600001,58800000,65100000,66299999];
        REC(7,:) = [125001,28800001,30000000,57600001,58800000,NaN(1,4)];
end
end

function ann=ann_reader_v02(annotation,Exp,from,to)
% ANNOTATION READER                                                  7/2025
% -------------------------------------------------------------------------
% This code is dedicated for reading of quality annotations of BUT QDB.
% Input: annotation - file (.csv)
%        Exp        - choice of expert(s) (1,2,3 or 4 ~ Exp1,Exp2,Exp3 or median consensus MC)
%        from       - from sample
%        to         - to sample
%
% Otput: variable "ann" - vector of annotations sample-by-sample
%
% Example of use: 
% ann = ann_reader_v02('100001_ANN.csv');                   -> size(ann) = [87087000, 4]
% ann = ann_reader_v02('100001_ANN.csv',4,1,1000);          -> size(ann) = [1000, 1]
% ann = ann_reader_v02('100001_ANN.csv',[1 4],101,500);     -> size(ann) = [400, 2]
% ann = ann_reader_v02([ID,'\',ID,'_ANN'],1:3,1001);        -> size(ann) = [87086000, 3]
%
% Created by authors of BUT QDB, 7/2025
% -------------------------------------------------------------------------

T = readtable([annotation,'.csv']);
TA = table2array(T);
sl = [1 4 7 10];
Lann = TA(end,sl(~isnan(TA(end,sl+1)))+1);

if nargin == 1
    Exp = 1:4;
    from = 1;
    to = Lann(1);
elseif nargin == 2
    from = 1;
    to = Lann(1);
elseif nargin == 3
    to = Lann(1);
end

ann = zeros(to,length(Exp));
poc=0; sl = sl(Exp);
for i = sl
    Lan=length(TA(~isnan(TA(:,i))));
    poc=poc+1;

    for j=1:Lan
        if  from >= TA(j,i) && from <= TA(j,i+1)
            M = j;
            break
        end
    end
    
    for j=M:Lan
        if to >= TA(j,i) && to <= TA(j,i+1)
            N = j;
            break
        end
    end
    
    for j=M:N
        ann(TA(j,i):TA(j,i+1),poc) = TA(j,i+2);
    end
end
ann = ann(from:to,:);
end

function [ConfM, ACC, E, F1, tim, Q123] = summarySTAT(ANN,Q,fs_ecg)

ConfM = zeros(3);                           % Confusion Matrix
for i = 1 : 3
    for j = 1 : 3
        ConfM(j,i) = sum(ANN==j & Q==i);
    end
end
classesANN = sum(ConfM,2);                  % # of samples in each group according to annotation
classesQ = sum(ConfM,1);                    % # of samples in each group according to algorithm 
all = sum(classesQ);                        % # of all samples 

TP = diag(ConfM);                           % # of true positiv cases in each group
FP = classesQ' - TP;                      	% # of false positiv cases in each group
FN = classesANN - TP;                    	% # of false negativ cases in each group

F1 =  2.*TP ./ (2.*TP + FP + FN) .* 100;    % F1 measure
ConfM = ConfM/all*100;                      % Confusion Matrix in percentage
ACC = sum(diag(ConfM));                     % Accuracy
E(1) = sum([ConfM(3,1:2), ConfM(2,1)]);     % Error I type  
E(2) = sum([ConfM(1,2:3), ConfM(2,3)]);     % Error II type  

tim = all/fs_ecg/60;                        % Time in minutes
Q123 = classesANN/all*100;                  % Q1, Q2, Q3 distribution in percentage
end

function SegAna = summarySEGS(ANN,Q,fs_ecg)

% -- Annotated segments (Actual) --
dANN = diff(ANN);       
seg_ann = diff([0; find(dANN~=0)]);
seg_ann(seg_ann==1) = [];
seg_ann = seg_ann/fs_ecg;       % segmants in seconds

% -- Detected segments (Predicted) --
dQ = diff(Q);
seg_q = diff([0; find(dQ~=0)]);
seg_q(seg_q==1) = [];       
seg_q = seg_q/fs_ecg;           % segmants in seconds

SegAna = [ length(seg_ann),	length(seg_q);...
     	max(seg_ann),       max(seg_q);...   
       	min(seg_ann),       min(seg_q);...
       	mean(seg_ann),      mean(seg_q);...
       	median(seg_ann),    median(seg_q)];
end

