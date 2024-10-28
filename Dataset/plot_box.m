clear all
clc
close all
set(0,'DefaultLineLineWidth',1);
set(0,'DefaultTextFontSize',20)
set(0,'DefaultTextInterpreter','latex')
set(0,'DefaultAxesFontSize',16)

load('bounding_boxes.mat');

vehicle = 1;
instant = 1;
number_actors = length(bounding_boxes{1, 1}.actors);
cmap = hsv(number_actors);  %# Creates a 6-by-3 set of colors from the HSV colormap

% take box corresponding to vehicle and instant
boxes = bounding_boxes{vehicle, instant}.boxes;
name = bounding_boxes{vehicle, instant}.actors;

for actor = 1 : number_actors
    
    drawBBox3D(boxes(:,:,actor), cmap(actor,:), regexprep(name{:,actor},'_','\\_'))

end 



