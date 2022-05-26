

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s4]=fig_e();
set(s4,'position',[0,.48,1,0.2]);

%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
[s5,s6,s7,s8,s9,s10] = fig_f();
set(s5,'position',[0.04+0,.41,0.6/6,0.08]);
set(s6,'position',[0.04+1/6,.41,0.6/6,0.08]);
set(s7,'position',[0.04+2/6,.41,0.6/6,0.08]);
set(s8,'position',[0.04+3/6,.41,0.6/6,0.08]);
set(s9,'position',[0.04+4/6,.41,0.6/6,0.08]);
set(s10,'position',[0.04+5/6,.41,0.6/6,0.08]);

%%






function [s1]=fig_e()
load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

s1 = subplot(1,24,4); hold on;
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

text(-750+sum(x_offset(1:ii+1)),1300,[num2str(zSlice(ii)),'nm<h <',num2str(zSlice(ii+1)),'nm'],'Color','w','FontSize',8);


if ii==12
    plot([sum(x_offset(1:ii+1))+600,sum(x_offset(1:ii+1))+600+400],[-1100,-1100],'Color','w','LineWidth',2);
end
end

end

function [s1,s2,s3,s4,s5,s6] = fig_f()

load('pixOL_com_loc_precision_NFP-700.mat');
load('est_retrieval_1.1_v15.mat');
load('data_for_DPPC_v15.mat');

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

