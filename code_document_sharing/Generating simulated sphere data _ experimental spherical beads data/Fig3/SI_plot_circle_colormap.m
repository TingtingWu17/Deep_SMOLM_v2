load('colorSpace.mat');
colorSpace = cat(2 , colorSpace,colorSpace);
figure('Color','k');
R = 0.015;
N = 16;
% because there are 31 points through the whole circle, 16 points at one direction, 16 points at opposite direction but one point is common (the middle point)
[rho,th] = meshgrid(linspace(0,R,N),linspace(0,2*pi,403));
xp = rho.*cos(th);
yp = rho.*sin(th);
Tp = squeeze(colorSpace);
pcolor(xp,yp,th/pi*180-180); hold on;
scatter(0,0,7000,[0,0,0],'filled');
shading('interp');
%colorbar
colormap(Tp)
axis off; axis image
set(gcf, 'InvertHardcopy', 'off');