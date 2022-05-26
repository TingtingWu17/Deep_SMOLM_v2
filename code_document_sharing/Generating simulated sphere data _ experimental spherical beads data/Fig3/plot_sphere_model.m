%saveas(gca,'Fig1.jpg','jpg')
%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);


hold on;

% coordinate for a plane
p1 = patch('XData',[0,0,0,0],'YData',[1000,-1000,-1000,1000]*1.1,'ZData',[-1000,-1000,1000,1000]*1.1);
p1.FaceColor = 	'#A2142F'; p1.FaceAlpha = 0.4;
% p2 = patch('XData',[1000,1000,-1000,-1000]*0.9,'YData',[-1000,1000,1000,-1000]*0.9,'ZData',[-1000,1000,1000,-1000]*0.9);
% p2.FaceColor = '#0072BD'; p2.FaceAlpha = 0.4; 
p2 = patch('XData',[1000,1000,-1000,-1000]*0.9,'YData',[1000,-1000,-1000,1000]*0.9,'ZData',[-1000,1000,1000,-1000]*0.9);
p2.FaceColor = '#0072BD'; p2.FaceAlpha = 0.4; 
p3 = patch('XData',[1000,1000,-1000,-1000],'YData',[-1000,1000,1000,-1000],'ZData',[0,0,0,0]);
p3.FaceColor = 'g'; p3.FaceAlpha = 0.7;   p3.FaceAlpha = 0.4; p3.EdgeColor = 'k';

% scatter3(-sqrt(1000^2-(1000-150)^2)*sqrt(2)/2,-sqrt(1000^2-(1000-150)^2)*sqrt(2)/2,-(1000-150),30,'filled','w');
% scatter3(-sqrt(1000^2-(1000-550)^2)*sqrt(2)/2,-sqrt(1000^2-(1000-550)^2)*sqrt(2)/2,-(1000-550),30,'filled','w');
% scatter3(-sqrt(1000^2-(1000-950)^2)*sqrt(2)/2,-sqrt(1000^2-(1000-950)^2)*sqrt(2)/2,-(1000-950),30,'filled','w');

[x,y,z] = sphere(100);
x = x*1000; y = y*1000; z = z*1000;
s = surface(x,y,z);
s.EdgeColor = 'none';
s.FaceColor = [148, 145, 148]/255;
s.FaceAlpha = 0.8;
axis image; %xlabel('x'); ylabel('y'); zlabel('h');
set(gca,'XTick',[]); set(gca,'YTick',[]); set(gca,'ZTick',[])
axis off;

quiver3(0,0,-1000,0,0,90,'AutoScaleFactor',25,'Color','k','LineWidth',2);
quiver3(0,0,-1000,90,0,0,'AutoScaleFactor',20,'Color','k','LineWidth',2);
quiver3(0,0,-1000,-90,0,0,'AutoScaleFactor',25,'Color','k','LineWidth',2,'LineStyle',':','ShowArrowHead','off');
quiver3(0,0,-1000,0,90,0,'AutoScaleFactor',15,'Color','k','LineWidth',2);
quiver3(0,0,-1000,0,-90,0,'AutoScaleFactor',15,'Color','k','LineWidth',2,'LineStyle',':','ShowArrowHead','off');
quiver3(0,0,-1000,0,-90,90,'AutoScaleFactor',15,'Color','r','LineWidth',2);
%%

r=1000;
x=0;
y=0;
th = linspace(0,2*pi,100);
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
zunit = zeros(size(xunit));


Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);

[x,y,z] = sphere(100);
x = x*1000; y = y*1000; z = z*1000;
s = surface(x,y,z);
s.EdgeColor = 'none';
s.FaceColor = 'k';
s.FaceAlpha = 1;
axis image; xlabel('x'); ylabel('y'); zlabel('h');
hold on;

Ry = [cosd(90),0,sind(90);
      0,1,0;
      -sind(90),0,cosd(90)]; 
plane1 = Ry*[xunit;yunit;zunit];
p1 = patch('XData',plane1(1,:)*1.3,'YData',plane1(2,:)*1.3,'ZData',plane1(3,:)*1.3);
p1.FaceColor = '#A2142F'; p1.FaceAlpha = 0.4; p1.EdgeColor = 'none';

plane2 = [xunit;yunit;zunit];
p2 = patch('XData',plane2(1,:)*1.3,'YData',plane2(2,:)*1.3,'ZData',plane2(3,:)*1.3);
p2.FaceColor = '#0072BD'; p2.FaceAlpha = 0.4; p2.EdgeColor = 'none';

Ry = [cosd(45),0,sind(45);
      0,1,0;
      -sind(45),0,cosd(45)]; 
plane3 = Ry*[xunit;yunit;zunit];
p3 = patch('XData',plane3(2,:)*1.3,'YData',plane3(1,:)*1.3,'ZData',-plane3(3,:)*1.3);
p3.FaceColor ='y'; p3.FaceAlpha = 0.4; p3.EdgeColor = 'none';
