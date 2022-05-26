%% for seeing the plaque area
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')

%% three viewing slices
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s2 = fig_viewing_3_slices1();

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s3 = fig_viewing_3_slices2();

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s4 = fig_viewing_3_slices3();

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
photon_thred = 1000; omega_thred = 0.000;
s11 = fig_z_thetaPer(photon_thred,omega_thred);

s12 = fig_rad_phi(photon_thred,omega_thred);

%%
photon_thred = 1000; omega_thred = 0.0000;
s1 = fig_signal_phi(photon_thred,omega_thred);

% %%
% photon_thred = 1000; omega_thred = 0.0000;
% s1 = fig_rad_h(photon_thred,omega_thred);

%%
photon_thred = 1000; omega_thred = 0.000;
 fig_omega_phi(photon_thred,omega_thred);

 %%
photon_thred = 1000; omega_thred = 0.000;
fig_omega_signal(photon_thred,omega_thred);
%%


function s1 = fig_viewing_3_slices1()

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

count = 0;

load('colorSpace.mat'); color = squeeze(colorSpace); % colormap
isColorbar = 0;
isScaleBar = 0;
% view 1: x-y    
zSlice = [900,1000];
indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<zSlice(2) & z_cord>zSlice(1) &SM_est_final(:,5)>1000;
phiD_cord(phiD_cord<0) = phiD_cord(phiD_cord<0)+180;

s1 = subplot(1,3,1);
isScaleBar = 0;
plot_scatter(x_cord(indx1),y_cord(indx1),phiD_cord(indx1), color,isColorbar,isScaleBar);
xlim([-1200,1200]);
ylim([-1200,1200]);

end

function s2 = fig_viewing_3_slices2()
load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


count = 0;

load('colorSpace.mat'); color = squeeze(colorSpace); % colormap
isColorbar = 0;
isScaleBar = 0;

% view 2: x-z    
% xSlice = [-100,100];
% indx2 = abs(x_cord)<1200 & abs(y_cord)<1200 & (x_cord-y_cord)<xSlice(2) & (x_cord-y_cord)>xSlice(1) & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>1000;
% locations = [x_cord(indx2),y_cord(indx2),z_cord(indx2)].';
% Rz = [cosd(45) -sind(45) 0;
%       sind(45) cosd(45) 0;
%       0 0 1];
% locationsRotate = Rz*locations;
% 
% s2 = subplot(1,3,2);
% isScaleBar = 1;
% plot_scatter(locationsRotate(2,:),locationsRotate(3,:),phiD_cord(indx2), color,isColorbar,isScaleBar);

xSlice = [-100,100];
indx2 = abs(x_cord)<1200 & abs(y_cord)<1200 & (x_cord)<xSlice(2) & (x_cord)>xSlice(1) & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>1000;

s2 = subplot(1,3,2);
isScaleBar = 0;
plot_scatter(y_cord(indx2),z_cord(indx2),phiD_cord(indx2), color,isColorbar,isScaleBar);

% set(gca,'ytick',[500,1000]);
% set(gca,'xtick',[-1000,0,1000]);
xlim([-1200,1200]);

end

function s3 = fig_viewing_3_slices3()
load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


count = 0;

load('colorSpace.mat'); color = squeeze(colorSpace); % colormap
isColorbar = 0;
isScaleBar = 0;

% view 3: x=y 
xzSlice = [-100,100];
indx3 = abs(x_cord)<1200 & abs(y_cord)<1200 & (z_cord+y_cord)<xzSlice(2)+1000 & (z_cord+y_cord)>xzSlice(1)+1000 & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>1000;

% indx1 = abs(x_cord)<1200 & abs(y_cord)<1200  & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>1000;
% figure();
% scatter3(x_cord(indx1),y_cord(indx1),z_cord(indx1),5,'filled','r'); xlabel('x'); ylabel('y'); zlabel('z'); 
% hold on; 
% scatter3(x_cord(indx3),y_cord(indx3),z_cord(indx3),5,'filled','b');
% axis image

s3 = subplot(1,3,3);
locations = [x_cord(indx3),y_cord(indx3),z_cord(indx3)].';
% Rx = [cosd(-45),0,sind(-45);
%       0,1,0;
%       -sind(-45),0,cosd(-45)];
Rx = [1,0,0;
      0,cosd(-45),-sind(-45);
      0,sind(-45),cosd(-45)];
locationsRotate = Rx*locations;
isScaleBar = 1;
plot_scatter(locationsRotate(1,:),locationsRotate(3,:),phiD_cord(indx3), color,isColorbar,isScaleBar); 
ylim([-400,max(locationsRotate(3,:))]);
xlim([-1200,1200]);
% set(gca,'ytick',[-200,400,1000]);
% set(gca,'xtick',[-1000,0,1000]);
end


function s1 = fig_z_thetaPer(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


theta_bins = linspace(0,90,25);
z_bins = linspace(0,1200,25);
%hotN = bin2Ddata(z_cord(indx1),theta_perpendicular(indx1),z_bins,theta_bins);
[hotN,theta_perpendicular_median] = bin2Ddata_ouput_median(z_cord(indx1),theta_perpendicular(indx1),z_bins,theta_bins);
s1 = subplot(1,2,2);
x_cord = (theta_bins(1:end-1)+theta_bins(2:end))/2;
y_cord = (z_bins(1:end-1)+z_bins(2:end))/2;
imagesc(x_cord,y_cord, hotN(1:end-1,1:end-1)); hold on;
plot(theta_perpendicular_median(1:end-1),y_cord,'LineWidth',2,'Color','r');
set(gca,'YDir','normal')
xlabel('\theta_\perp (\circ)','FontSize',10); ylabel('h (nm)','FontSize',10);
colorbar;
end




function s1 = fig_rad_phi(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


phi_bins = linspace(-180,180,50);
s1 = subplot(1,2,1);
hotN = bin2Ddata(phiD_cord(indx1),phiD_ref(indx1),phi_bins,phi_bins);
x_cord = (phi_bins(1:end-1)+phi_bins(2:end))/2;
imagesc(x_cord,x_cord, hotN(1:end-1,1:end-1)); 
set(gca,'YDir','normal')
xlabel('\phi_{sphere} (\circ)','FontSize',10); ylabel('\phi (\circ)','FontSize',10);
colorbar;
end


function s1 = fig_rad_h(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


phi_bins = linspace(-180,180,50);
z_bins = linspace(0,1200,25);
%hotN = bin2Ddata(z_cord(indx1),theta_perpendicular(indx1),z_bins,theta_bins);
s1 = subplot(1,2,1);
hotN = bin2Ddata(z_cord(indx1),phiD_cord(indx1),z_bins,phi_bins);
x_cord = (phi_bins(1:end-1)+phi_bins(2:end))/2;
y_cord = (z_bins(1:end-1)+z_bins(2:end))/2;
imagesc(x_cord,y_cord, hotN(1:end-1,1:end-1)); 
set(gca,'YDir','normal')
xlabel('\phi_{sphere} (\circ)','FontSize',10); ylabel('h (nm)','FontSize',10);
colorbar;
end

function s1 = fig_signal_phi(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


phi_bins = linspace(-180,180,50);
signal_bins = linspace(1000,10000,20);
s1 = figure();
hotN = bin2Ddata(phiD_cord(indx1),SM_est_final(indx1,5),phi_bins,signal_bins);
hotN = hotN./max(hotN,[],1);
y_cord = (phi_bins(1:end-1)+phi_bins(2:end))/2;
x_cord = (signal_bins(1:end-1)+signal_bins(2:end))/2;
imagesc(x_cord,y_cord, hotN(1:end-1,1:end-1)); 
set(gca,'YDir','normal')
xlabel('signal (photons)','FontSize',10); ylabel('\phi (\circ)','FontSize',10);
colorbar;
end


function fig_omega_phi(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>300 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


phi_bins = linspace(-180,180,50);
omega_bins = linspace(0,2*pi,10);
figure();
hotN = bin2Ddata(phiD_cord(indx1),omega(indx1),phi_bins,omega_bins);
hotN = hotN./max(hotN,[],1);
%hotN = hotN;
y_cord = (phi_bins(1:end-1)+phi_bins(2:end))/2;
x_cord = (omega_bins(1:end-1)+omega_bins(2:end))/2;
imagesc(x_cord/pi,y_cord, hotN(1:end-1,1:end-1)); 
set(gca,'YDir','normal')
xlabel('\Omega (\pi sr)','FontSize',10); ylabel('\phi (\circ)','FontSize',10);
colorbar;
end



function s1 = fig_omega_signal(photon_thred,omega_thred)

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>000 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);


omega_bins = linspace(0,2*pi,10);
signal_bins = linspace(1000,10000,20);
s1 = figure();
hotN = bin2Ddata(omega(indx1),SM_est_final(indx1,5),omega_bins,signal_bins);
%hotN = hotN;
hotN = hotN./max(hotN,[],1);
y_cord = (omega_bins(1:end-1)+omega_bins(2:end))/2;
x_cord = (signal_bins(1:end-1)+signal_bins(2:end))/2;
imagesc(x_cord,y_cord/pi, hotN(1:end-1,1:end-1)); 
set(gca,'YDir','normal')
xlabel('signal (photons)','FontSize',10); ylabel('\Omega (\pi sr)','FontSize',10);
colorbar;
end


function plot_scatter(x_cord,y_cord,colorPara, colormapC,isColorbar,isScaleBar)
scatter(x_cord,y_cord,3,colorPara,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5); axis image; 
colormap(colormapC);
whitebg('k'); %#ok<WHITEBG>
ax = gca;
ax.XColor = 'k'; ax.YColor = 'k';
set(gca,'Color','k')
ax.FontSize = 10; 
set(gca,'xtick',[]);
set(gca,'ytick',[]);
if isColorbar==1
    hcb = colorbar('Color','k');
    %hcb.Title.String = "\theta(\circ)";
end
if isScaleBar == 1
   hold on;
   plot([max(x_cord)-500,max(x_cord)-500+400],[min(y_cord)+230,min(y_cord)+230],'Color','w','LineWidth',2);
    
end
%viscircles([0,0],1000)
%xlabel('x','Color','k'); ylabel('y','Color','k'); 
axis image
end



