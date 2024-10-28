clear all
clc
close all
set(0,'DefaultLineLineWidth',1);
set(0,'DefaultTextFontSize',20)
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)


% load('bounding_boxes.mat');
% load('true_ped_boxes.mat');
% for l = 1 : size(bounding_boxes,1)
%     for k = 1 : size(bounding_boxes,2)
%         bounding_boxes{l,k}.boxes = true_ped_boxes(l,k).boxes;
%         bounding_boxes{l,k}.actors = true_ped_boxes(l,k).names;
% 
%         for box = 1 : size(bounding_boxes{l,k}.boxes,3)
%             if ~ isempty(bounding_boxes{l,k}.boxes(:,:,box)) 
%                 bounding_boxes{l,k}.boxes(:,:,box) = rotate_point(bounding_boxes{l,k}.boxes(:,:,box)); 
%             end
%         end
%     end
% end
% new_dataset = bounding_boxes;
% save('true_ped_boxes_rotated.mat', 'new_dataset')


load('bounding_boxes.mat');
load('bounding_boxes_noise_Pointpillars.mat');
for l = 1 : size(bounding_boxes,1)
    for k = 1 : size(bounding_boxes,2)
        bounding_boxes{l,k}.boxes = new_dataset{l,k}.boxes;
        bounding_boxes{l,k}.actors = new_dataset{l,k}.actors;

        for box = 1 : size(bounding_boxes{l,k}.boxes,3)
            if ~ isempty(bounding_boxes{l,k}.boxes(:,:,box)) 
                bounding_boxes{l,k}.boxes(:,:,box) = rotate_point(bounding_boxes{l,k}.boxes(:,:,box)); 
            end
        end
    end
end
new_dataset = bounding_boxes;
save('bounding_boxes_noise_Pointpillars_rotated.mat', 'new_dataset')



function point = rotate_point (point) 
    
    % Select ccorner with highest x value
    base = point(:,1:4);
    [m, i] = max(base(1,:) + base(2,:));
    % [m, j] = max(base(2,:));

    % Rotate until first corner has highest x
    while i ~= 1
        % Rotate clockwise
        base = point(:,1:4);
        up_ = point(:,5:8);
        base = circshift(base,1,2);
        up_ = circshift(up_,1,2);
        % point(:,1:4) = circshift(point(:,1:4),1,2);
        % point(:,5:8) = circshift(point(:,5:8),1,2);
        [m, i] = max(base(1,:) + base(2,:));
        % [m, j] = max(base(2,:));
        point = [base,up_];
    end
 
end






