%% for seeing the plaque area
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')




%% 
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s5,s6,s7,s8,s9,s10] = fig_f();
set(s5,'position',[0.04+0,.41,0.6/6,0.08]);
set(s6,'position',[0.04+1/6,.41,0.6/6,0.08]);
set(s7,'position',[0.04+2/6,.41,0.6/6,0.08]);
set(s8,'position',[0.04+3/6,.41,0.6/6,0.08]);
set(s9,'position',[0.04+4/6,.41,0.6/6,0.08]);
set(s10,'position',[0.04+5/6,.41,0.6/6,0.08]);

%% three viewing slices
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s2 = fig_viewing_3_slices1();

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s3 = fig_viewing_3_slices2();
%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s4 = fig_viewing_3_slices3();

%% omega of bleb

Fig1 = figure('Units','inches','InnerPosition',[1,1,2.3,1.8]);   
s1 = plot_fig_bleb();

set(s1,'position',[0.15,0.17,0.8,0.8]);

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
photon_thred = 1000; omega_thred = 0.0;
s11 = fig_z_thetaPer(photon_thred,omega_thred);

s12 = fig_rad_phi(photon_thred,omega_thred);

%%


function s1 = plot_fig_bleb()

load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

r_plq = [200,250,250]; center_plq = [-347,873,650];
indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;
%indx_plq = data_plaque(:,1);
indx_plq = x_cord>(center_plq(1)-r_plq(1)) & x_cord<(center_plq(1)+r_plq(1))...
           & y_cord>(center_plq(2)-r_plq(2)) & y_cord<(center_plq(2)+r_plq(2))...
           & z_cord>(center_plq(3)-r_plq(3)) & z_cord<(center_plq(3)+r_plq(3))...
           &SM_est_final(:,5)>1000;
           
       
edges = linspace(0,2,30);       
s1 = subplot(1,1,1);
histogram(omega(indx)/pi,edges,'Normalization','probability','FaceColor','#77AC30');  hold on;
histogram(omega(indx_plq)/pi,edges,'Normalization','probability','FaceColor','#7E2F8E');
whitebg('w');
xlim([0,2]);

xlabel('\Omega_{\pi}');  ylabel('relative occurance');
legend('all SMs','bleb');

ax = gca;
ax.FontSize = 10; 

end


function [s1,s2,s3,s4,s5,s6] = fig_f()

load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

z_cord = z_cord+25;
sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);

zSlice = linspace(000,1200,13); d_slice = zSlice(2)-zSlice(1); % plotted slice 


count = 0;
for ii=2:2:length(zSlice)-1
    count = count+1;
    
indx = abs(x_cord)<1300 & abs(y_cord)<1300 & z_cord<zSlice(ii+1) & z_cord>zSlice(ii) &SM_est_final(:,5)>1000;
r_data = sqrt((x_cord(indx).^2+y_cord(indx).^2));

%------------  reference sphere  ----------------
R_cur = sqrt(1000^2-(1000- zSlice(ii)-d_slice/2).^2);
R_top = sqrt(1000^2-(1000- zSlice(ii)-d_slice).^2);
R_bottom = sqrt(1000^2-(1000- zSlice(ii)).^2);
if zSlice(ii)+d_slice<=0
    if zSlice(ii)+d_slice/2<=0
        R_cur=0;
    end
    R_bottom=0;
end

if zSlice(ii)>=1000
   temp1 = max([R_top,R_bottom,R_cur]);
   temp2 = min([R_top,R_bottom,R_cur]);
   R_top = temp1;
   R_bottom = temp2;
end
%------------------------------
[~,indx_top]=min(abs(zSlice(ii+1)-z_range*10^9));
sigma_top = sigma_r(indx_top)*10^9;

[~,indx_bottom]=min(abs(zSlice(ii)-z_range*10^9));
sigma_bottom = sigma_r(indx_bottom)*10^9;

s_temp = subplot(1,24,count+4);  hold on;

N_bins = length(r_data)/20;
edges=linspace((prctile(r_data,2)-50),(prctile(r_data,98)+50),N_bins);
edges_distance = edges(2)-edges(1);
h = histogram(r_data,edges);
h.FaceColor = [128, 255, 0]/255;
%h.FaceColor = [0.4660, 0.6740, 0.1880];
h.EdgeColor = 'none';

%-------------calculate the FWHM -----------------

pdf_data = h.Values; center_data = h.BinEdges;
[max_Data,indx_max] = max(pdf_data);
half_max= max_Data/2;
[~,indx_left] = min(abs(pdf_data(1:indx_max)-half_max));
[~,indx_right] = min(abs(pdf_data(indx_max+1:end)-half_max));
data_FWHM(ii) = center_data(indx_right+indx_max)-center_data(indx_left);


x_ax = R_bottom-180:R_top+180;
if ii==12
   x_ax =  R_bottom-270:R_top+270;
end
gauss_pdf = normpdf((1:length(x_ax))-length(x_ax)/2,0,(sigma_top+sigma_bottom)/2);
circ_pdf = zeros(size(x_ax));
circ_pdf = (x_ax+1)./cos(asin((x_ax+1)/1000))-(x_ax)./cos(asin((x_ax)/1000));
circ_pdf(x_ax<(R_bottom)|x_ax>(R_top)) = 0; 
circ_pdf = real(circ_pdf);
circ_pdf(circ_pdf<0) = 0;
if sum(circ_pdf)==0
   pdf = gauss_pdf; pdf = pdf/sum(pdf);
else
circ_pdf = circ_pdf/sum(circ_pdf);
pdf = conv(circ_pdf,gauss_pdf,'same');
pdf = pdf/sum(pdf);
end
pdf = pdf*max(pdf_data)/max(pdf);
plot(x_ax,pdf,'Color',[255 255 153]/255, 'LineWidth',1.5);
%plot(x_ax,pdf*sum(indx)*edges_distance,'Color',[0.9290, 0.6940, 0.1250], 'LineWidth',1.5);
xlabel('r (nm)'); ylabel('count');
set(gca,'xcolor','w') 
set(gca,'ycolor','w') 
ax = gca;
ax.FontSize = 8;
xlim([min(x_ax(1),edges(1)),max(x_ax(end),edges(end))]);

%-------------calculate the FWHM -----------------

%
%loc_data_distr = deconv(pdf_data/sum(pdf_data),gauss_pdf);
pdf_temp = circ_pdf*sum(indx)*edges_distance;
[max_Data,indx_max] = max(pdf_temp);
half_max= max_Data/2;
[~,indx_left] = min(abs(pdf_temp(1:indx_max)-half_max));
[~,indx_right] = min(abs(pdf_temp(indx_max+1:end)-half_max));
pfd_FWHM_sphere(ii) =indx_right+indx_max-indx_left;

%
pdf_temp = pdf;
[max_Data,indx_max] = max(pdf_temp);
half_max= max_Data/2;
[~,indx_left] = min(abs(pdf_temp(1:indx_max)-half_max));
[~,indx_right] = min(abs(pdf_temp(indx_max+1:end)-half_max));
pfd_FWHM_whole(ii) = indx_right+indx_max-indx_left;



% set(gcf,'Color','none');
% whitebg('none');

if count==1
s1=s_temp;
elseif count==2
    s2 = s_temp;
elseif count==3
    s3 = s_temp;
elseif count==4
    s4 = s_temp;
elseif count==5
    s5 = s_temp;
elseif count==6
    s6 = s_temp;    
end

end

end

function s1 = fig_viewing_3_slices1()

load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);


count = 0;

load('colorSpace.mat'); color = squeeze(colorSpace); % colormap
%color = color(1:round(size(color,1)/2),:);
isColorbar = 0;
isScaleBar = 0;
% view 1: x-y    
zSlice = [900,1000];
indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<zSlice(2) & z_cord>zSlice(1) &SM_est_final(:,5)>1000;
phiD_cord(phiD_cord<0) = phiD_cord(phiD_cord<0)+180;

s1 = subplot(1,3,1);
isScaleBar = 0;
plot_scatter(x_cord(indx1),y_cord(indx1),phiD_cord(indx1), color,isColorbar,isScaleBar);
xlim([-1200,1200])
ylim([-1200,1200])
end

function s2 = fig_viewing_3_slices2()

load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

z_cord = z_cord;
sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);


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
xlim([-1200,1200])
% set(gca,'ytick',[500,1000]);
% set(gca,'xtick',[-1000,0,1000]);
end

function s3 = fig_viewing_3_slices3()
load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

 z_cord = z_cord;
sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);


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
xlim([-1200,1200])

% set(gca,'ytick',[-200,400,1000]);
% set(gca,'xtick',[-1000,0,1000]);

end


function s1 = fig_z_thetaPer(photon_thred, omega_thred)

load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
xlabel('\theta_\perp (\circ)'); ylabel('h (nm)');

theta_bins = linspace(0,90,25);
z_bins = linspace(0,1200,25);
[hotN,theta_perpendicular_median] = bin2Ddata_ouput_median(z_cord(indx1),theta_perpendicular(indx1),z_bins,theta_bins);
s1 = subplot(1,2,2);
imagesc(theta_bins(1:end-1),z_bins(1:end-1), hotN(1:end-1,1:end-1)); hold on;
plot(theta_perpendicular_median(1:end-1),z_bins(1:end-1),'LineWidth',2,'Color','r');
set(gca,'YDir','normal')
xlabel('\theta_\perp (\circ)','FontSize',10); ylabel('h (nm)','FontSize',10);
colorbar;
end

function s1 = fig_rad_phi(photon_thred,omega_thred)

load('pixOL_com_loc_precision_NFP-700.mat');
load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');


indx1 = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<800 & z_cord>200 &SM_est_final(:,5)>photon_thred &omega>omega_thred;

%s1 = scatter(theta_perpendicular(indx1),z_cord(indx1),1,'filled','MarkerFaceAlpha',0.5,'MarkerEdgeAlpha',0.5);
xlabel('\theta_\perp (\circ)'); ylabel('h (nm)');

phi_bins = linspace(-180,180,50);
subplot(1,2,1);
hotN = bin2Ddata(phiD_cord(indx1),phiD_ref(indx1),phi_bins,phi_bins);
s1 = imagesc(phi_bins(1:end-1),phi_bins(1:end-1), hotN(1:end-1,1:end-1));
set(gca,'YDir','normal')
xlabel('\phi_{sphere} (\circ)','FontSize',10); ylabel('\phi (\circ)','FontSize',10);
colorbar;
end

function s1 = fig_h()
s1 = subplot(1,24,11); hold on; box on;


load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');

zSlice = linspace(000,1200,13);
count = 0;
for ii = 2:1:length(zSlice)-1
    count = count+1;
%Y{count} = [num2str(zSlice(ii)),'nm<z<',num2str(zSlice(ii+1)),'nm'];
Y{count} = [num2str(zSlice(ii)+50)];
end
X = categorical(Y);
X = reordercats(X,Y);
X = [150:100:1150];
plot(X,sigma_DPPC,'LineWidth',1.5,'Color',[0, 0.4470, 0.7410]); hold on;


load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


plot(X,sigma_DPPC,'LineWidth',1.5,'Color',[0.9290, 0.6940, 0.1250]); hold on;
plot(X,sigma_perfect,'LineWidth',1.5,'Color',[0.8500, 0.3250, 0.0980]);
%legend('DPPC+40%chol','DPPC','theory','EdgeColor','none','Color','none','FontSize',8); ylabel('FWHM(nm)');
set(gcf,'Color','w');
whitebg('w');
xlim([150,1150]);
xlabel('h(nm)');
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
   plot([max(x_cord)-400,max(x_cord)-400+400],[min(y_cord)+400,min(y_cord)+400],'Color','w','LineWidth',2);
    
end
%viscircles([0,0],1000)
%xlabel('x','Color','k'); ylabel('y','Color','k'); 
axis image
end