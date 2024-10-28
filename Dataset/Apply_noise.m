clear all
clc
close all
set(0,'DefaultLineLineWidth',1);
set(0,'DefaultTextFontSize',20)
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)




for sigma = 0.1:0.2:2

    load('bounding_boxes.mat');
    new_dataset = bounding_boxes;

%     sigma = 0.5; % m
    
    for vehicle=1:size(bounding_boxes,1)
        for instant=1:size(bounding_boxes,2)
            boxes = bounding_boxes{vehicle, instant}.boxes;
            for actor = 1 : length(bounding_boxes{vehicle, instant}.actors)
                
                old_coord = boxes(:,:,actor);
                
                noise = sigma*randn(3,1);
                
                new_coord = old_coord + noise;
                
                new_dataset{vehicle, instant}.boxes(:,:,actor) = new_coord;
                   
            end
            
        end
    end
    
    save(sprintf('bounding_boxes_noise_%0.1f.mat',sigma), 'new_dataset')

end











