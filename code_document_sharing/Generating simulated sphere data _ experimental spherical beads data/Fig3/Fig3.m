%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
%exportgraphics(Fig1,'BarChart.png','Resolution',1000)
%% DPPC+chol xy view

Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s1] = fig_b(); %[left, bottom, width, height] D=space for label

set(s1,'position',[0.125,.70,0.125*2,0.125]);
%% DPPC+chol xz view
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s2 = fig_c(); %[left, bottom, width, height] D=space for label
set(s2,'position',[0.13+0.13*2+0.03,.70, 0.13*2,0.13*2]);


%% DPPC x-y view
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
s3 = fig_d(); %[left, bottom, width, height] D=space for label
set(s3,'position',[0.13+0.13*4+0.06,.70,0.13*2,0.13*2]);

%% scatter at each z-slices
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s4]=fig_e();
set(s4,'position',[0,.48,1,0.2]);

%% histogram of r distribution
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s5,s6,s7,s8,s9,s10] = fig_f();
set(s5,'position',[0.04+0,.41,0.6/6,0.08]);
set(s6,'position',[0.04+1/6,.41,0.6/6,0.08]);
set(s7,'position',[0.04+2/6,.41,0.6/6,0.08]);
set(s8,'position',[0.04+3/6,.41,0.6/6,0.08]);
set(s9,'position',[0.04+4/6,.41,0.6/6,0.08]);
set(s10,'position',[0.04+5/6,.41,0.6/6,0.08]);

%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')


% % PSF matching at three axial locations
% 
% Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
% 
% [s13,s14,s15,s16,s17,s18,s19,s20,s21,s22,s23,s24]=fig_g();
% 
% a = 0.075;
% set(s13,'position',[0,0.28,a,a]);
% set(s14,'position',[a+0.002,0.28,a,a]);
% 
% set(s15,'position',[a*2+0.01,0.28,a,a]);
% set(s16,'position',[a*3+0.01+0.002,0.28,a,a]);
% 
% set(s17,'position',[a*4+0.01*2,0.28,a,a]);
% set(s18,'position',[a*5+0.01*2+0.002,0.28,a,a]);
% 
% set(s19,'position',[0,0.20,a,a]);
% set(s20,'position',[a+0.002,0.20,a,a]);
% 
% set(s21,'position',[a*2+0.01,0.20,a,a]);
% set(s22,'position',[a*3+0.01+0.002,0.20,a,a]);
% 
% set(s23,'position',[a*4+0.01*2,0.20,a,a]);
% set(s24,'position',[a*5+0.01*2+0.002,0.20,a,a]);
%% all last conlumn figures
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.9,1.6]); 
textInclude = 1;
fakeplot = 1;
s1 = fig_i(textInclude,fakeplot);


% bled

s2 = plot_fig_bleb();


% FWHM plot
s4 = fig_h();



% spherical shape demonstration
s3 = fig_r_vs_h();
set(s1,'position',[0.05,0.17,0.78/4,0.75]);
set(s2,'position',[0.05+0.26,0.17,0.78/4,0.75]);
set(s3,'position',[0.05+0.51,0.17,0.78/4,0.75]);
set(s4,'position',[0.15+0.65,0.17,0.78/4,0.75]);
%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.9,1.6]); 
textInclude = 0;
fakeplot = 0;
s1 = fig_i(textInclude,fakeplot);
set(s1,'position',[0.05,0.17,0.78/4,0.75]);

%%
function [s1]=fig_b()
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;

isColorbar=0; isScaleBar=0;
s1=subplot(1,24,1);
plot_scatter(x_cord(indx),z_cord(indx),thetaD(indx), turbo(256),isColorbar,isScaleBar);



end

function [s2]=fig_c()
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;

isColorbar=0; isScaleBar=0;
plot_scatter(x_cord(indx),z_cord(indx),thetaD(indx), turbo(256),isColorbar,isScaleBar);
s2=subplot(1,24,2);
plot_scatter(x_cord(indx),y_cord(indx),thetaD(indx), turbo(256),isColorbar,isScaleBar);



end

function [s3]=fig_d()
load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>2000;
isColorbar=0; isScaleBar=1;
s3=subplot(1,24,3);
plot_scatter(x_cord(indx),y_cord(indx),thetaD(indx), turbo(256),isColorbar,isScaleBar);


end


function [s1]=fig_e()
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')


s1 = subplot(2,24,4); hold on;
%-----------------------------plotting parameters-----------------------------
r = 70; %line length
zSlice = linspace(000,1200,13); d_slice = zSlice(2)-zSlice(1); % plotted slice 
x_offset = [0,sqrt(1000^2-(1000-(zSlice(2:end))))*1.2]; %the distance between each circle plot
load('colorSpace.mat'); color = squeeze(colorSpace); % colormap
phiD_dic = linspace(-180,180,403); %partition the colormap with phi
%-----------------------------

for ii=2:2:length(zSlice)-1
indx = abs(x_cord)<1300 & abs(y_cord)<1300 & z_cord<zSlice(ii+1) & z_cord>zSlice(ii) &SM_est_final(:,5)>1000;

for jj=1:length(indx)
    if indx(jj)==1
        [~,color_indx] = min(abs(phiD_cord(jj)-phiD_dic));
        plot(sum(x_offset(1:ii+1))+[x_cord(jj)-r*sind(thetaD(jj)).*cosd(phiD_cord(jj)),x_cord(jj),x_cord(jj)+r*sind(thetaD(jj)).*cosd(phiD_cord(jj))].',...
             [y_cord(jj)-r*sind(thetaD(jj)).*sind(phiD_cord(jj)),y_cord(jj),y_cord(jj)+r*sind(thetaD(jj)).*sind(phiD_cord(jj))].',...
             'Color',color(color_indx,:),'LineWidth',0.5);
    end
end
axis image
%xlim([-1300,1300]);ylim([-1300,1300]);

%-------- plot reference sphere-----------------------------
% R_cur = sqrt(1000^2-(1000- zSlice(ii)-d_slice/2).^2);
% R_top = sqrt(1000^2-(1000- zSlice(ii)-d_slice).^2);
% R_bottom = sqrt(1000^2-(1000- zSlice(ii)).^2);
% if zSlice(ii)+d_slice<=0
%     if zSlice(ii)+d_slice/2<=0
%         R_cur=0;
%     end
%     R_bottom=0;
% end

%-------- figure setting-----------------------------
set(gca,'xtick',[]);
set(gca,'ytick',[]);
ax = gca;
ax.FontSize = 10; 
axis off;
set(gcf, 'InvertHardcopy', 'off');
whitebg('k');
set(gcf,'Color',[0 0 0]);

%text(-750+sum(x_offset(1:ii+1)),1300,[num2str(zSlice(ii)),'nm<h <',num2str(zSlice(ii+1)),'nm'],'Color','w','FontSize',8);


if ii==12
    plot([sum(x_offset(1:ii+1))+600,sum(x_offset(1:ii+1))+600+400],[-1100,-1100],'Color','w','LineWidth',2);
end
end

end

function [s1,s2,s3,s4,s5,s6] = fig_f()

load('pixOL_com_loc_precision_NFP-700.mat');
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

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
pd = fitdist(r_data,'Normal');
hold on; 
%plot(edges,edges_distance*length(r_data)*1/(sqrt(2*pi)*pd.sigma)*exp(-1/2*((edges-pd.mu)/pd.sigma).^2),'LineWidth',2)
%-------------calculate the FWHM -----------------

%pdf_data = h.Values; center_data = h.BinEdges;
pdf_data = edges_distance*length(r_data)*1/(sqrt(2*pi)*pd.sigma)*exp(-1/2*((edges-pd.mu)/pd.sigma).^2); 
center_data = h.BinEdges;
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

function s1 = fig_h()



load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')
zSlice = linspace(000,1200,13);
count = 0;



X = [150:100:1150];
s1 = subplot(1,4,3); hold on; box on;
plot(X,sigma_DPPC(1:end),'LineWidth',1.5,'Color',[0, 0.4470, 0.7410]); hold on;

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


plot(X,sigma_DPPC(1:end),'LineWidth',1.5,'Color',[0.9290, 0.6940, 0.1250]); hold on;
plot(X,sigma_perfect(1:end),'LineWidth',1.5,'Color',[0.8500, 0.3250, 0.0980]);
%legend('DPPC+40%chol','DPPC','theory','EdgeColor','none','Color','none','FontSize',8); ylabel('FWHM(nm)');
set(gcf,'Color','w');
whitebg('w');
xlim([150,1150]);
xlabel('h(nm)');
end


function s1 = fig_h_v2()
s1 = subplot(1,24,11); hold on; box on;

load('pixOL_com_loc_precision_NFP-700.mat');
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

[sigma_perfect, sigma_DPPC_chol]=calculateFWHM(sigma_x_opt,sigma_y_opt,x_cord,y_cord,z_cord,SM_est_final,z_range);


X = [150:100:1150];
plot(X,sigma_DPPC_chol,'LineWidth',1.5,'Color',[0, 0.4470, 0.7410]); hold on;

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');


[sigma_perfect, sigma_DPPC]=calculateFWHM(sigma_x_opt,sigma_y_opt,x_cord,y_cord,z_cord,SM_est_final,z_range);
plot(X,sigma_DPPC,'LineWidth',1.5,'Color',[0.9290, 0.6940, 0.1250]); hold on;
plot(X,sigma_perfect,'LineWidth',1.5,'Color',[0.8500, 0.3250, 0.0980]);
%legend('DPPC+40%chol','DPPC','theory','EdgeColor','none','Color','none','FontSize',8); ylabel('FWHM(nm)');
set(gcf,'Color','w');
whitebg('w');
xlim([150,1150]);
xlabel('h(nm)');

ax = gca;
ax.FontSize = 9; 
end

function s1 = fig_i(textInclude,fakeplot)
s1 = subplot(1,4,1); hold on;

if fakeplot==0

load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;
dens = 0.7;
sizeS = 0.5;
hold on;
scatter(theta_perpendicular(indx),omega(indx)/pi,sizeS,'filled','MarkerFaceColor',	[0.9290, 0.6940, 0.1250],'MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
%scatter(theta_perpendicular(indx),omega(indx)/pi,sizeS,'filled','MarkerFaceColor',	[0.8500, 0.3250, 0.0980],'MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
if textInclude==1
ylabel('\Omega(\pi)'); xlabel('\theta\perp  (\circ)');
end
hold on;
%scatter(median(theta_perpendicular(indx)),median(omega(indx))/pi,100,'+','MarkerFaceColor',[0.8500, 0.3250, 0.0980],'MarkerEdgeColor',[0.8500, 0.3250, 0.0980],'LineWidth',2);
ax = gca;
ax.FontSize = 9; 
box on;
whitebg('w');
set(gcf,'Color','w');
xticks([0,45,90])
xlim([0,90]);



load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')
indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;
dens = 0.5;
%figure();
scatter(theta_perpendicular(indx),omega(indx)/pi,sizeS,'filled','MarkerFaceColor',[0, 0.4470, 0.7410],'MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

hold on;
scatter(median(theta_perpendicular(indx)),median(omega(indx))/pi,100,'+','MarkerFaceColor',[0, 102, 204]./255,'MarkerEdgeColor',[0, 102, 204]./255,'LineWidth',2);

ax = gca;
ax.FontSize =9; 
box on;
whitebg('w');
set(gcf,'Color','w');


load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;
%scatter(median(theta_perpendicular(indx)),median(omega(indx))/pi,100,'+','MarkerFaceColor',[0.8500, 0.3250, 0.0980],'MarkerEdgeColor',[0.8500, 0.3250, 0.0980],'LineWidth',2);
scatter(median(theta_perpendicular(indx)),median(omega(indx))/pi,100,'+','MarkerFaceColor',[173, 150, 38]/255,'MarkerEdgeColor',[173, 150, 38]/255,'LineWidth',2);
end

end

function [s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12]=fig_g()

load('classify_PSF_DPPC_40chol_v27.mat');
load('simulate_PSF_pixOL_com.mat');
axiOFFs = 'off';
isScaleBar = 'off';

s1 = subplot(1,24,13);
plotIx(I_exp_z150,axiOFFs,isScaleBar);
s2 = subplot(1,24,14);
plotIy(I_exp_z150,axiOFFs,isScaleBar);

s3 = subplot(1,24,15);
plotIx(I_exp_z550,axiOFFs,isScaleBar);
s4 = subplot(1,24,16);
plotIy(I_exp_z550,axiOFFs,isScaleBar);

s5 = subplot(1,24,17);
plotIx(I_exp_z950,axiOFFs,isScaleBar);
s6 = subplot(1,24,18);
plotIy(I_exp_z950,axiOFFs,isScaleBar);

s7 = subplot(1,24,19);
plotIx(I_sim_z150,axiOFFs,isScaleBar);
s8 = subplot(1,24,20);
plotIy(I_sim_z150,axiOFFs,isScaleBar);

s9 = subplot(1,24,21);
plotIx(I_sim_z550,axiOFFs,isScaleBar);
s10 = subplot(1,24,22);
plotIy(I_sim_z550,axiOFFs,isScaleBar);

s11 = subplot(1,24,23);
plotIx(I_sim_z950,axiOFFs,isScaleBar);
isScaleBar = 'on';
s12 = subplot(1,24,24);
plotIy(I_sim_z950,axiOFFs,isScaleBar);

end


function s2 = plot_fig_bleb()

load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

%r_plq = [200,250,250]; center_plq = [-347,873,700];
r_plq = [200,250,250]; center_plq = [-380,968,700];
indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1000 & z_cord>00 &SM_est_final(:,5)>1000;
%indx_plq = data_plaque(:,1);
indx_plq = x_cord>(center_plq(1)-r_plq(1)) & x_cord<(center_plq(1)+r_plq(1))...
           & y_cord>(center_plq(2)-r_plq(2)) & y_cord<(center_plq(2)+r_plq(2))...
           & z_cord>(center_plq(3)-r_plq(3)) & z_cord<(center_plq(3)+r_plq(3))...
           &SM_est_final(:,5)>1000;
           
       
       
s2 = subplot(1,4,2);
histogram(theta_perpendicular(indx),20,'Normalization','probability','FaceColor','#77AC30');  hold on;
histogram(theta_perpendicular(indx_plq),20,'Normalization','probability','FaceColor','#7E2F8E');
whitebg('w');


xlabel('\theta_{\perp}');  ylabel('relative occurance');
legend('all SMs','defect');

ax = gca;
ax.FontSize = 9; 
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
   plot([max(x_cord)-500,max(x_cord)-500+400],[min(y_cord)+150,min(y_cord)+150],'Color','w','LineWidth',2);
    
end
%viscircles([0,0],1000)
%xlabel('x','Color','k'); ylabel('y','Color','k'); 

end



function plotIx(I_raw,axisoff,isScaleBar)
Imax1 = max(I_raw,[],'all');
temp = hot(300); map1(:,1) = min(temp(:,3),1); 
map1(:,2) = min(temp(:,1)*0.50+temp(:,2)*0.50,1);
map1(:,3) = min(temp(:,1)*1,1);

imgSz = size(I_raw,1);

Ix = I_raw(:,1:imgSz); Iy = I_raw(:,imgSz+1:imgSz*2);
imagesc([Ix]);  axis image; caxis([0,Imax1]);   colormap(gca,'hot');  %colorbar('Location','north'); 
if strcmp(axisoff, 'off')    
    axis off;
end


if strcmp(isScaleBar, 'on')
line([imgSz-8.5470-4,imgSz-4],[imgSz-5 imgSz-5],'Color',[254, 254, 254]./255,'LineWidth',2);
end


end

function plotIy(I_raw,axisoff,isScaleBar)
Imax1 = max(I_raw,[],'all');
temp = hot(300); map1(:,1) = min(temp(:,3),1); 
map1(:,2) = min(temp(:,1)*0.50+temp(:,2)*0.50,1);
map1(:,3) = min(temp(:,1)*1,1);

imgSz = size(I_raw,1);

Ix = I_raw(:,1:imgSz); Iy = I_raw(:,imgSz+1:imgSz*2);
imagesc([Iy]);  axis image; caxis([0,Imax1]);   colormap(gca,map1);  %colorbar('Location','north'); 
if strcmp(axisoff, 'off')    
    axis off;
end


if strcmp(isScaleBar, 'on')
line([imgSz-8.5470-4,imgSz-4],[imgSz-5 imgSz-5],'Color',[254, 254, 254]./255,'LineWidth',2);
end


end


function [sigma_perfect, sigma_DPPC] = calculateFWHM(sigma_x_opt,sigma_y_opt,x_cord,y_cord,z_cord,SM_est_final,z_range)
% load('pixOL_com_loc_precision_NFP-700.mat');
% load('est_xyCentered_v28.mat')
% load('data_for_DPPC_chol_v28.mat')

sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);
zSlice = linspace(000,1200,13); d_slice = zSlice(2)-zSlice(1); % plotted slice 

count = 0;
for ii=1:1:length(zSlice)-1
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



N_bins = length(r_data)/20;
edges=linspace((prctile(r_data,2)-50),(prctile(r_data,98)+50),N_bins);
edges_distance = edges(2)-edges(1);
pd = fitdist(r_data,'Normal');
%-------------calculate the FWHM -----------------

%pdf_data = h.Values; center_data = h.BinEdges;
X = edges;
pdf_data = edges_distance*length(r_data)*1/(sqrt(2*pi)*pd.sigma)*exp(-1/2*((X-pd.mu)/pd.sigma).^2); 
center_data = edges;
[max_Data,indx_max] = max(pdf_data);
half_max= max_Data/2;
[~,indx_left] = min(abs(pdf_data(1:indx_max)-half_max));
[~,indx_right] = min(abs(pdf_data(indx_max+1:end)-half_max));
data_FWHM(ii) = center_data(indx_right+indx_max)-center_data(indx_left);


x_ax = R_bottom-180:R_top+180;
if ii==12
   x_ax = R_bottom-270:R_top+270;
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

end

sigma_DPPC = (data_FWHM(2:1:end));
sigma_DPPC(sigma_DPPC<0)=0;
sigma_perfect = (pfd_FWHM_whole(2:1:end));
end



function [s24]=fig_r_vs_h()
load('est_xyCentered_v28.mat')
load('data_for_DPPC_chol_v28.mat')

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>00 &SM_est_final(:,5)>1000;

z_slice = linspace(00,1200,20);
z_slice_center = (z_slice(1:end-1)+z_slice(2:end))/2;
for ii = 1:length(z_slice)-1
    
indx1 = indx & z_cord>=z_slice(ii) & z_cord<z_slice(ii+1);
r = (sqrt(x_cord(indx1).^2+y_cord(indx1).^2));
fig=figure; set(fig,'visible','off');
H=histogram(r,15);
h=H.Values;
[~,indx_peak] = max(h);
r_mean_chol(ii) = (H.BinEdges(indx_peak)+H.BinEdges(indx_peak+1))/2;
close(fig);
%r_mean(ii) = median(sqrt(x_cord(indx1).^2+y_cord(indx1).^2));
z_cur = (z_slice(ii)+z_slice(ii+1))/2;
z_cur = linspace(z_slice(ii),z_slice(ii+1),100);
r_theory_cur = mean(sqrt(1000^2-(1000-z_cur).^2));
fig=figure; set(fig,'visible','off');
H=histogram(r_theory_cur,15);
h=H.Values;
[~,indx_peak] = max(h);
r_theory(ii) = (H.BinEdges(indx_peak)+H.BinEdges(indx_peak+1))/2;
close(fig);

  
end

s24 = subplot(1,4,4);
plot(z_slice_center,r_mean_chol,'LineWidth',2); hold on;
plot(z_slice_center,r_theory,'LineWidth',2);


load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

indx = abs(x_cord)<1200 & abs(y_cord)<1200 & z_cord<1200 & z_cord>00 &SM_est_final(:,5)>1000;
for ii = 1:length(z_slice)-1
    
indx1 = indx & z_cord>=z_slice(ii) & z_cord<z_slice(ii+1);
r = (sqrt(x_cord(indx1).^2+y_cord(indx1).^2));
fig=figure; set(fig,'visible','off');
H=histogram(r,15);
h=H.Values;
[~,indx_peak] = max(h);
r_mean(ii) = (H.BinEdges(indx_peak)+H.BinEdges(indx_peak+1))/2;
close(fig);

%r_mean(ii) = median(sqrt(x_cord(indx1).^2+y_cord(indx1).^2));
z_cur = (z_slice(ii)+z_slice(ii+1))/2;
z_cur = linspace(z_slice(ii),z_slice(ii+1),100);
r_theory(ii) = mean(sqrt(1000^2-(1000-z_cur).^2));
  
end
plot(z_slice_center,r_mean,'LineWidth',2); 
xlabel('h(nm)'); ylabel('r');
legend('DPPC+40%chol','ideal','DPPC');
ax = gca;
ax.FontSize = 9; 
ylim([min(r_theory),max([r_mean,r_mean_chol,r_theory])]);
xlim([z_slice(1),z_slice(end-1)]);
end

