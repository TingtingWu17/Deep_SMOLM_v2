
%load('est_xyCentered_v28.mat');
load('est_retrieval_1.1_v15.mat');
%
load('colorSpace.mat');
%load('est_xy_centered_v13.mat');


x_cord = x; y_cord = y;z_cord = z;  %-90

mux = sind(phiD); muy = cosd(phiD); 
phiD_cord = atan2(mux,-muy)/pi*180;

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>1000;

%plot 3D spikes
figure('Color',[0.1,0.1,0.1]);

hold on

%load('turbo.mat');
%colormap(turbo);
%color = squeeze(colorSpace);
color = turbo(256);
thetaD_dic = linspace(0,90,256);
phiD_dic = linspace(-180,180,403);

count = 0;
r=50;
 for ii=1:length(indx)
    if indx(ii)==1
        count = count+1;
        if rem(count,3)==0
        [~,color_indx] = min(abs(thetaD(ii)-thetaD_dic));
        plot3([x_cord(ii)-r*sind(thetaD(ii)).*cosd(phiD_cord(ii)),x_cord(ii),x_cord(ii)+r*sind(thetaD(ii)).*cosd(phiD_cord(ii))].',...
             [y_cord(ii)-r*sind(thetaD(ii)).*sind(phiD_cord(ii)),y_cord(ii),y_cord(ii)+r*sind(thetaD(ii)).*sind(phiD_cord(ii))].',...
             [z_cord(ii)+r*cosd(thetaD(ii)),z_cord(ii),z_cord(ii)-r*cosd(thetaD(ii))].',...
             'Color',color(color_indx,:),'LineWidth',1);
        end
    end
end
axis image

%title('DPPC+chol','Color','w','FontSize',13);
annotation('textbox',[0.42,0.8,0.1,0.1],'String','DPPC-only','Color','w','FontSize',13)
xlabel('x(nm)'); ylabel('y(nm)'); zlabel('h(nm)');
colormap(color); caxis([0,90]); hcb = colorbar('Position',[0.9,0.3,0.03,0.3]);  title(hcb,'\theta(\circ)','Color','w','FontSize',13); 
ax = gca;
ax.FontSize = 10; 
whitebg('black');
set(gcf, 'InvertHardcopy', 'off');
phi_view = linspace(0,180,50);
theta_view = linspace(0,20,50);
%----------save image at different viewing angles---------------------
 for jj=1:length(theta_view)%1:length(theta_view)
     xPlane = [-1 1 1 -1]*1300;      % X coordinates of plane corners, ordered around the plane
    yPlane1 = [0,0,0,0];      % Corresponding y coordinates for plane 1
    zPlane = [0 0 2100 3500];  % Z coordinates of plane corners
    hold on;                   % Add to existing plot
    h = patch(xPlane, yPlane1, zPlane, 'k', 'FaceAlpha', 0.8);  % Plot plane 1
    rotate(h,[0,0,1],phi_view(jj));
    rotate(h,[1,0,0],360-theta_view(jj));

     view(phi_view(jj),theta_view(jj));
     zlim([0,1500]);
     if jj>=37
        zlim([0,1500+(jj-36)*10]);
     end
     if jj>=45
        zlim([0,1500+(jj-36)*10+13*(jj-44)]);
     end
     set(gcf, 'Color', 'None')
     axis vis3d;
     saveas(gcf,['C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\figure_generation_data\Fig3\spikes_DPPC\',num2str(jj),'.png'])
     delete(h);
 end

%% z slice scattering plot
zSlice = linspace(-50,1200,50);

for ii=1:length(zSlice)
indx = abs(x_cord)<1300 & abs(y_cord)<1300 & z_cord<zSlice(ii)+100 & z_cord>zSlice(ii) &SM_est_final(:,5)>1000;
space = round(linspace(1,6380,4500));

figure('Color',[0.1,0.1,0.1]);
hold on

color = squeeze(colorSpace);
thetaD_dic = linspace(0,90,256);
phiD_dic = linspace(-180,180,403);
r = 40;
%caxis([-180,180]);
hold on;
%colorbar;
for jj=1:length(indx)
    if indx(jj)==1
        [~,color_indx] = min(abs(phiD_cord(jj)-phiD_dic));
        plot([x_cord(jj)-r*sind(thetaD(jj)).*cosd(phiD_cord(jj)),x_cord(jj),x_cord(jj)+r*sind(thetaD(jj)).*cosd(phiD_cord(jj))].',...
             [y_cord(jj)-r*sind(thetaD(jj)).*sind(phiD_cord(jj)),y_cord(jj),y_cord(jj)+r*sind(thetaD(jj)).*sind(phiD_cord(jj))].',...
             'Color',color(color_indx,:),'LineWidth',1);
    end
end
axis image


%---
whitebg('k');
%colorbar('Color','w');
%xlabel('x','Color','w'); ylabel('y','Color','w'); 
set(gca,'xtick',[]);
set(gca,'ytick',[]);
ax = gca;
ax.FontSize = 10; 
axis off;



set(gcf,'Color',[0 0 0]);
text(-530,1300,[num2str(round(zSlice(ii))),'nm<h <',num2str(round((zSlice(ii)+100))),'nm'],'Color','w','FontSize',13);
plot([-1000,-1200],[-1000,-1000],'Color','w','LineWidth',2); hold on;
text(-1300,-910,'200 nm','FontSize',12)
title('DPPC+chol','Color','w','FontSize',13);
%set(gcf, 'InvertHardcopy', 'off');

%phi colorbar
center = [1100,-900];
R = 150;
N = 16;
% because there are 31 points through the whole circle, 16 points at one direction, 16 points at opposite direction but one point is common (the middle point)
[rho,th] = meshgrid(linspace(0,R,N),linspace(0,2*pi,403));
xp = rho.*cos(th);
yp = rho.*sin(th);
Tp = squeeze(colorSpace);
pcolor(xp+center(1),yp+center(2),th/pi*180-180); hold on;
scatter(0+center(1),0+center(2),200,[0,0,0],'filled');
shading('interp');
text(center(1)+(R+20)*cosd(0),center(2)+(R+20)*sind(0),'\phi=0^\circ','FontSize',12);
%text(center(1)+(R+20)*cosd(45)+10,center(2)+(R+20)*sind(45)+10,'45^\circ','FontSize',12);
text(center(1)+(R+20)*cosd(90)-70,center(2)+(R+20)*sind(90)+50,'90^\circ','FontSize',12);
%text(center(1)+(R+20)*cosd(135)-200,center(2)+(R+20)*sind(135)+20,'135^\circ','FontSize',12);
%text(center(1)+(R+20)*cosd(-45),center(2)+(R+20)*sind(-45),'-45^\circ','FontSize',12);
text(center(1)+(R+20)*cosd(-90)-100,center(2)+(R+20)*sind(-90)-50,'-90^\circ','FontSize',12);
%text(center(1)+(R+20)*cosd(-135)-250,center(2)+(R+20)*sind(-135)-20,'-135^\circ','FontSize',12);
text(center(1)+(R+20)*cosd(180)-250,center(2)+(R+20)*sind(180),'180^\circ','FontSize',12);
%colorbar
colormap(Tp)
axis off; axis image
set(gcf, 'InvertHardcopy', 'off');
xlim([-1300,1300]);ylim([-1300,1400]);
saveas(gcf,['C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\figure_generation_data\Fig3\spikes-plane\',num2str(ii),'.png'])
close all

end


%%

load('colorSpace.mat');
figure('Color','k');

