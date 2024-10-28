clear all
clc
close all
load(('bounding_boxes_noise_Pointpillars.mat'));
noise_dataset = new_dataset;
load('true_ped_boxes_adapted_format.mat');
true_dataset = new_dataset;

instant = 564;
vehicle = 7;
actor = 1;

% new_dataset{vehicle,instant}.boxes(:,:,actor)
% plot_points(instant, vehicle , actor, new_dataset)
% noise_dataset{vehicle,instant}.boxes(:,:,actor) = rotate_point(noise_dataset{vehicle,instant}.boxes(:,:,actor)); 
% new_dataset{vehicle,instant}.boxes(:,:,actor)
% true_name = noise_dataset{vehicle, instant}.actors{actor};


true_name = true_dataset{vehicle, instant}.actors{actor};

for actor2 = 1 : length(noise_dataset{vehicle, instant}.actors)

    noise_name = noise_dataset{vehicle, instant}.actors{actor2};
    if strcmp(noise_name, true_name)
        figure
        grid
        hold on
        title('true no corrections')
        plot_points(instant, vehicle , actor, true_dataset)
        figure
        grid
        hold on
        title('noise no corrections')
        plot_points(instant, vehicle , actor2, noise_dataset)

        true_dataset{vehicle,instant}.boxes(:,:,actor) = rotate_point(true_dataset{vehicle,instant}.boxes(:,:,actor));
        noise_dataset{vehicle,instant}.boxes(:,:,actor2) = rotate_point(noise_dataset{vehicle,instant}.boxes(:,:,actor2));

        figure
        grid
        hold on
        title('true si corrections')
        plot_points(instant, vehicle , actor, true_dataset)
        figure
        grid
        hold on
        title('noise si corrections')
        plot_points(instant, vehicle , actor2, noise_dataset)

        true_coord = true_dataset{vehicle,instant}.boxes(:,:,actor);
        noise_coord = noise_dataset{vehicle,instant}.boxes(:,:,actor2);   
        err = true_coord- noise_coord

    end
end









function plot_points (instant, vehicle, actor, dataset)
    points = 1:8;
    for point=points
        x = dataset{vehicle, instant}.boxes(1,point,actor);
        y = dataset{vehicle, instant}.boxes(2,point,actor);
        scatter(x,y)
        if point>=5
            text( x + 0.1, y+ 0.1, ...
                sprintf('%d', point), 'Color', 'r', 'FontSize', 22);
        else
            text( x, y, ...
                sprintf('%d', point), 'Color', 'r', 'FontSize', 22);
        end
    end
end




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
    

%     % Rotate until first corner has highest y
%     base = point(:,1:4);
%     [m, i] = max(base(2,:));
%     while i ~= 1
%         % Rotate clockwise
%         base = point(:,1:4);
%         up_ = point(:,5:8);
%         base = circshift(base,1,2);
%         up_ = circshift(up_,1,2);
%         % point(:,1:4) = circshift(point(:,1:4),1,2);
%         % point(:,5:8) = circshift(point(:,5:8),1,2);
%         [m, i] = max(base(2,:));
%         point = [base,up_];
%     end
end





