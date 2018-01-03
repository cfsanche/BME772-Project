function performanceMetrics = validation(seizure_auto, seizureGT)

%     figure; %plot vs gt 
%     plot(seizureMarker_auto); hold on; plot(seizureMarker_gt);
%     title('Outcome'); ylabel('Classification');
%     legend('Results','Gold Standard');
    
%     w=256;% 1 second window
%     num_seg = floor(length(seizureGT)/w);
%     seizure_auto=reshape(seizure_auto(1:num_seg*w),w,num_seg);%reshapes column wise
%     seizureGT=reshape(seizureGT(1:num_seg*w),w,num_seg);
%     true_detec = 0;
%     for i=1:num_seg
%         if(seizure_auto(:,i)==seizureGT(:,i))
%             true_detec=true_detec+1;
%         end
%     end
%     acc2 = (true_detec)/num_seg*100;
%   
%     performanceMetrics = acc2;
    
     TP = length(find(seizureGT == 1 & seizure_auto == 1)); 
     FP = length(find(seizureGT == 0 & seizure_auto == 1)); 
     FN = length(find(seizureGT == 1 & seizure_auto == 0)); 
     TN = length(find(seizureGT == 0 & seizure_auto == 0)); 
     SEN = TP/(TP+FN)*100 % recall 
     SPE = TN/(TN+FP)*100 
     ACC = (TP+TN)/(TP+TN+FP+FN)*100
     precision = TP/(TP+FP)*100
     F1score = 2.*TP/(2.*TP + FP + FN)*100   
end
