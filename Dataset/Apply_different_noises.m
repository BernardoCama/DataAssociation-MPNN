clear all
clc
close all
set(0,'DefaultLineLineWidth',1);
set(0,'DefaultTextFontSize',20)
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)

% Noises
% https://openaccess.thecvf.com/content/ICCV2021/supplemental/Luo_Score-Based_Point_Cloud_ICCV_2021_supplemental.pdf

% 1 Isotropic Gaussian Noise sigma 0.5
sigma_iso = 0.5;
% 2 Non-Isotropic Gaussian Noise sigma (multivariate normal distribution)
sigma_non_iso = sigma_iso^2 * [1   -1/2   -1/4; 
                            -1/2  1     -1/4;
                            -1/4  -1/4   1  ];
% 3 Laplacian Noise std deviation 0.5
sigma_iso = 0.5;
% 4 Uniform Noise
sigma_iso = 0.5;
% 5 Discrete Noise
distribution = [ 0 0 0;
                 0 0 0;
                 0 0 0;
                 0 0 0;
                 sigma_iso 0 0;
                 -sigma_iso 0 0;
                 0 sigma_iso 0;
                 0 -sigma_iso 0;
                 0 0 sigma_iso;
                 0 0 -sigma_iso];

% noise = sigma_iso*randn(3,1);
% noise = mvnrnd([0 0 0],sigma_non_iso);
% noise = laprnd(3, 1, 0, sigma_iso);
% noise = rand_pick_sphere(1,0,sigma_iso,0,0,0);
% noise = datasample(distribution,1);



noises = ["Iso_gaussian","Non_Iso_gaussian","Laplacian", "Uniform","Discrete"];
noises = ["Non_Iso_gaussian","Laplacian", "Uniform","Discrete"];

num_noises = length(noises);


for i = 1:num_noises

    load('bounding_boxes.mat');
    new_dataset = bounding_boxes;
%     load('true_ped_boxes.mat');
%     new_dataset = true_ped_boxes;
    
    
    for vehicle=1:size(bounding_boxes,1)
        for instant=1:size(bounding_boxes,2)
            boxes = bounding_boxes{vehicle, instant}.boxes;
            for actor = 1 : length(bounding_boxes{vehicle, instant}.actors)
                
                old_coord = boxes(:,:,actor);
                
                noise = compute_noise (noises{i}, sigma_iso, sigma_non_iso, distribution);
                
                new_coord = old_coord + noise;
                
                new_dataset{vehicle, instant}.boxes(:,:,actor) = new_coord;
                   
            end
            
        end
    end
    
    save(sprintf('bounding_boxes_noise_%s.mat',noises{i}), 'new_dataset')

end


function noise = compute_noise (type, sigma_iso, sigma_non_iso, distribution)
    
    if type == "Iso_gaussian"
        noise = sigma_iso*randn(3,1);
    elseif type == "Non_Iso_gaussian"
        noise = mvnrnd([0 0 0],sigma_non_iso)';
    elseif type == "Laplacian"
        noise = laprnd(3, 1, 0, sigma_iso);
    elseif type == "Uniform"
        noise = rand_pick_sphere(1,0,sigma_iso,0,0,0);
    elseif type == "Discrete"
        noise = datasample(distribution,1)';
    end
end






