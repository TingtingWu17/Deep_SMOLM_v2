%%
f_forwardModel = @(x,G_re,b) abs((G_re*x));
f_loss = @(Iobs,Iest) sum(Iest-Iobs.*log(Iest+10^-16));


imgPara.img_sizex=101;
imgPara.img_sizey=101;


frame = 0;
for ii=1:50
   
   SM_est_cur = [];
   while isempty(SM_est_cur)
       frame = frame+1;
        SM_est_cur = SM_est_save_all(SM_est_save_all(:,1)==frame,:);
   end
   [gamma,loc] = SM_est2gamma(SM_est_cur,imgPara);
   N = size(loc,2);
   gamma = reshape(gamma,[],1);
   b = reshape(SM_bkg(:,:,frame),[],1);
    [G_re,loc_re_new,~] = update_basisMatrix(N,gamma,loc,imgPara);
    I = f_forwardModel(gamma,G_re,b);
    I_rec=reshape(I,101,202)+2;
    %plotI(I_rec,18);
    %saveas(gca,['C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\3Dbeads_experiment\20210414\reconst_image\' num2str(ii),',jpg'],'jpg');
    I_raw=reshape(SM_img(:,:,frame)*toPhoton,101,202);
    plotI(I_raw,I_rec,20,20);
    plotI(zeros(101,202),zeros(101,202),20,20);
    saveas(gca,['C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\3Dbeads_experiment\20210414\raw_img\' num2str(ii),',jpg'],'jpg');
    close all;
    

end


function plotI(I_raw,I_rec,Imax1,Imax2)
temp = hot(300); map1(:,1) = min(temp(:,3),1); 
map1(:,2) = min(temp(:,1)*0.50+temp(:,2)*0.50,1);
map1(:,3) = min(temp(:,1)*1,1);

imgSz = 101;
Fig = figure('Color',[0.1,0.1,0.1]);

set(Fig, 'Units','centimeters','InnerPosition', [12 12 4 8]);
I = I_raw;
%set(Fig, 'Units','centimeters','InnerPosition', [8 8 4 4]);

Ix = I(:,1:imgSz); Iy = I(:,imgSz+1:imgSz*2);
%Ix= imgaussfilt(Ix,1); Iy = imgaussfilt(Iy,1);
ax1 = axes('Position',[0.1 0.5 .4 .4]);
%ax2 = axes('Position',[0.5 0.2 .4 .4]); 
imagesc([Ix]);  axis image; caxis([0,Imax1]);  axis off; colormap(ax1,'hot');  colorbar('Location','north');
%ax1 = axes('Position',[0.1 0.2 .4 .4]);
ax2 = axes('Position',[0.5 0.5 .4 .4]); 
imagesc([Iy]); axis image
axis off; axis image;  caxis([0,Imax1]);  colormap(ax2,map1); colorbar('Location','north');
line([0 0],[0 imgSz],'Color',[196, 192, 192]./255,'LineWidth',2); 
line([imgSz-10.2564*1.667-4,imgSz-4],[imgSz-5 imgSz-5],'Color',[254, 254, 254]./255,'LineWidth',2);
%set(gcf, 'InvertHardcopy', 'off');


I = I_rec;
%set(Fig, 'Units','centimeters','InnerPosition', [8 8 4 4]);
%set(Fig, 'Units','centimeters','InnerPosition', [12 12 4 4]);
Ix = I(:,1:imgSz); Iy = I(:,imgSz+1:imgSz*2);
%Ix= imgaussfilt(Ix,1); Iy = imgaussfilt(Iy,1);
ax1 = axes('Position',[0.1 0.2 .4 .4]);
%ax2 = axes('Position',[0.5 0.2 .4 .4]); 
imagesc([Ix]);  axis image; caxis([0,Imax2]);  axis off; colormap(ax1,'hot');  colorbar;
%ax1 = axes('Position',[0.1 0.2 .4 .4]);
ax2 = axes('Position',[0.5 0.2 .4 .4]); 
imagesc([Iy]); axis image
axis off; axis image;  caxis([0,Imax2]);  colormap(ax2,map1);
line([0 0],[0 imgSz],'Color',[196, 192, 192]./255,'LineWidth',2); 
line([imgSz-10.2564*1.667-4,imgSz-4],[imgSz-5 imgSz-5],'Color',[254, 254, 254]./255,'LineWidth',2);
set(gcf, 'InvertHardcopy', 'off');
end
