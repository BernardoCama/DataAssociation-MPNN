function drawBBox3D(box,color,name)


hold on
box = box';

%Bottom
plot3([box(1,1) box(2,1)],[box(1,2) box(2,2)],[box(1,3) box(2,3)],'LineWidth',1.5,'Color',color,'DisplayName', name);
plot3([box(2,1) box(3,1)],[box(2,2) box(3,2)],[box(2,3) box(3,3)],'LineWidth',1.5,'Color',color);
plot3([box(3,1) box(4,1)],[box(3,2) box(4,2)],[box(3,3) box(4,3)],'LineWidth',1.5,'Color',color);
plot3([box(1,1) box(4,1)],[box(1,2) box(4,2)],[box(1,3) box(4,3)],'LineWidth',1.5,'Color',color);

%Top
plot3([box(5,1) box(6,1)],[box(5,2) box(6,2)],[box(5,3) box(6,3)],'LineWidth',1.5,'Color',color);
plot3([box(6,1) box(7,1)],[box(6,2) box(7,2)],[box(6,3) box(7,3)],'LineWidth',1.5,'Color',color);
plot3([box(7,1) box(8,1)],[box(7,2) box(8,2)],[box(7,3) box(8,3)],'LineWidth',1.5,'Color',color);
plot3([box(5,1) box(8,1)],[box(5,2) box(8,2)],[box(5,3) box(8,3)],'LineWidth',1.5,'Color',color);

%Top-bottom
plot3([box(1,1) box(5,1)],[box(1,2) box(5,2)],[box(1,3) box(5,3)],'LineWidth',1.5,'Color',color);
plot3([box(2,1) box(6,1)],[box(2,2) box(6,2)],[box(2,3) box(6,3)],'LineWidth',1.5,'Color',color);
plot3([box(3,1) box(7,1)],[box(3,2) box(7,2)],[box(3,3) box(7,3)],'LineWidth',1.5,'Color',color);
plot3([box(4,1) box(8,1)],[box(4,2) box(8,2)],[box(4,3) box(8,3)],'LineWidth',1.5,'Color',color);

% legend([plt1])
text(mean(box(:,1)),mean(box(:,2)),mean(box(:,3)),name,'FontSize',30)


end

