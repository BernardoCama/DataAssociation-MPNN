clear all
clc
close all
set(0,'DefaultLineLineWidth',1);
set(0,'DefaultTextFontSize',20)
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)

% Convert from struct to cell, and name the dataset as new_dataset

% load('bounding_boxes.mat');
% load('estimated_ped_boxes_withoutFP.mat');
% for l = 1 : size(bounding_boxes,1)
%     for k = 1 : size(bounding_boxes,2)
%         bounding_boxes{l,k}.boxes = estimated_ped_boxes_withoutFP(l,k).boxes;
%         bounding_boxes{l,k}.actors = estimated_ped_boxes_withoutFP(l,k).names';
%     end
% end
% 
% new_dataset = bounding_boxes;
% save(sprintf('bounding_boxes_noise_%s.mat','Pointpillars'), 'new_dataset')



% load('bounding_boxes.mat');
% load('estimated_ped_boxes.mat');
% for l = 1 : size(bounding_boxes,1)
%     for k = 1 : size(bounding_boxes,2)
%         bounding_boxes{l,k}.boxes = estimated_ped_boxes(l,k).boxes;
%         bounding_boxes{l,k}.actors = estimated_ped_boxes(l,k).names';
%     end
% end
% 
% new_dataset = bounding_boxes;
% save(sprintf('bounding_boxes_noise_%s.mat','Pointpillars_with_FP'), 'new_dataset')



load('bounding_boxes.mat');
load('true_ped_boxes.mat');
for l = 1 : size(bounding_boxes,1)
    for k = 1 : size(bounding_boxes,2)
        bounding_boxes{l,k}.boxes = true_ped_boxes(l,k).boxes;
        bounding_boxes{l,k}.actors = true_ped_boxes(l,k).names';
    end
end

new_dataset = bounding_boxes;
save(sprintf('true_ped_boxes_%s.mat','adapted_format'), 'new_dataset')


