%% Necessary information

% place following files in a directory called source_dir
% 1- transmissionRatio_name: a mat file that contains transmit_ratio_L2R as
% a variable
% transmit_ratio_L2R is a scalar indicating the light transmission ratio
% between left and right channels
%
% 2- zerothOrder_name: a mat file containing zerothOrder as a variable
% zerothOrder_RL is a 1*2 array specifying the zeroth order factor
% in the pupil plane according to [x_channel y_channel] or [right channel left
% channel] formt
%
% 3- image stack and background stack in binary format
% You may want to write your image and background stack into binary format
% with names img_stack_name.bin and backg_stack_name.bin accordingly
% use writeSMLMbackg2bin function

%% simulate images
imgSize = 61;
fileName = 'F:\data\20210410 3D lipid+ATTO647N\offset subtracted\31-35\';
SMLMName = [fileName,'34_SM1.tif'];
Nimg = 1000;

offsetR = Tiff(SMLMName,'r');
for i=1:Nimg
    setDirectory(offsetR,i);
    SM_img(:,:,i) = double(offsetR.read);

end

SM_img = [SM_img(:,1:imgSize,:),fliplr(SM_img(:,1+imgSize:imgSize*2,:))];
%%

fileName =  'F:\data\20210410 3D lipid+ATTO647N\offset subtracted\31-35\';
SMLMName = [fileName,'31_SM1_bkg.tif'];
Nimg = 100;

offsetR = Tiff(SMLMName,'r');

setDirectory(offsetR,1);
SM_bkg(:,:) = double(offsetR.read);
SM_bkg = [SM_bkg(:,1:imgSize,:),fliplr(SM_bkg(:,1+imgSize:imgSize*2,:))];
%% format data and computational resources

%----------------------------------------------
num_frames = 1400; % total number of frames in your stack
num_frames_backg = 1400; % number of background frames; it can be equal to ONE frame or the same as num_frames
num_frames_p = 100; % number of frames to be analyzed in parallel by all workers
number_of_workers = 4; % number of workers
imgSizeH = 570; 
imgSizeW = 2048;
%imgCenter = [262,369];
%imgCenterX = [302,369]; 
%imgCenterY = [302,369]; 
imgCenterX = [7.21*10^4,1.85*10^4]/58.5;  %in unit of pixel
imgCenterY = ([4.80*10^4,2.05*10^4]+[0,25])/58.5;  %in unit of pixel
imgCenterX(:) = imgCenterX(end:-1:1);
imgCenterY(:) = imgCenterY(end:-1:1);
imgSize = size(SM_img,1);
%imgSize = 357;
toPhoton = 1/3.4683;


%% analyze via RoSEO

%----------------------------------------------
% create nanoscope object analysis

maskName = fullfile('phasemask', 'pixOL_v12');
emitter_wavelength = 610; %nm
refractiveIndx=1.515;% objective refractive index
sampleRefractiveIndx=1.314;% sample refractive index

left_to_right_trans_ratio = 1.1449; %transmit_ratio_L2R; for Di03-R488/561,523/610 bandpass; y channel.x channel
%1.5350;%transmit_ratio_L2R; for Di03-R488/561, 593/46; y channel.x channel
zerothOrder = [0,0];%zerothOrder_RL;
%construct phasemaskpara
phasemaskpara.zeroorder = zerothOrder;
phasemaskpara.maskname = maskName;

imgPara.pix_sizex = 58.5; %the griding unit for RoSEO3D, in unit of nm
imgPara.pix_sizez = 50;
imgPara.number_axial_pixels = 24;
imgPara.axial_grid_points = [-4:1:19]*imgPara.pix_sizez;

%%
n1 = Nanoscope('imageSize', imgSize*3+6,...
    'ADcount', 1/3.4683,...
    'emissWavelength', emitter_wavelength, ...
    'refractiveIndx',refractiveIndx,...
    'sampleRefractiveIndx',sampleRefractiveIndx,...
    'phasemaskpara', phasemaskpara);
% create PSF matrix accounting for channel transmission ratio

%
[PSFx, PSFy] = n1.createPSFstruct3D(n1,...
    'ytoxchanneltransratio', left_to_right_trans_ratio,...
    'normal_focal_plane',-200,...
    'axial_grid_points',imgPara.axial_grid_points); %distance is is nm
%%
n1 = Nanoscope('imageSize', imgSize,...
    'ADcount', 1/3.4683,...
    'emissWavelength', emitter_wavelength, ...
    'refractiveIndx',refractiveIndx,...
    'sampleRefractiveIndx',sampleRefractiveIndx,...
    'phasemaskpara', phasemaskpara);

[FPSFx, FPSFy] = n1.createPSFstruct(n1,...
    'ytoxchanneltransratio', left_to_right_trans_ratio,...
    'normal_focal_plane',-200,...
    'molecule_plane',[350,0]);




%% define image parameters

Bx = cat(4,PSFx.XXx,PSFx.YYx,PSFx.ZZx,PSFx.XYx,PSFx.XZx,PSFx.YZx,...
     PSFx.XXxdx,PSFx.YYxdx,PSFx.ZZxdx,...
     PSFx.XXxdy,PSFx.YYxdy,PSFx.ZZxdy,...
     PSFx.XXxdz,PSFx.YYxdz,PSFx.ZZxdz);
By = cat(4,PSFy.XXy,PSFy.YYy,PSFy.ZZy,PSFy.XYy,PSFy.XZy,PSFy.YZy,...
     PSFy.XXydx,PSFy.YYydx,PSFy.ZZydx,...
     PSFy.XXydy,PSFy.YYydy,PSFy.ZZydy,...
     PSFy.XXydz,PSFy.YYydz,PSFy.ZZydz);

imgPara.img_size = imgSize;
imgPara.PSF_size_opt = 21; % the best size for croping the PSF (pixels)
imgPara.PSF_size = imgPara.PSF_size_opt;
imgPara.Bx = Bx;
imgPara.By = By;
imgPara.maxNumBeads = 5; %the maximum number of beads in a image frame
imgPara.PSF_size_opt_min = 11; % the minimum size for croping the PSF
%%
%initialize a parallel pool
p = gcp('nocreate');
if isempty(p)
    parpool('local', 8)
end
%%
PSFx_isotropic = (1/3*PSFx.XXx(:,:,3)+1/3*PSFx.YYx(:,:,3)+1/3*PSFx.ZZx(:,:,3));
PSFy_isotropic = (1/3*PSFy.XXy(:,:,3)+1/3*PSFy.YYy(:,:,3)+1/3*PSFy.ZZy(:,:,3));
%count1 = 0;
backg = SM_bkg*toPhoton;
%backg = [ones(57,57)*1.8374,ones(57,57)*2.3385];
parfor kk = 1:1000
SMLM_img = SM_img(:,:,kk)*toPhoton;
%SMLM_img = [SM_img([1:57]+2,[1:57]+2,kk),SM_img([1:57]+2+1,[1:57]+61+2-1,kk)]*toPhoton;
%
SM_est = RoSEO3D(SMLM_img, backg,n1,FPSFx, FPSFy,imgPara,'regval',.22);

SM_est_save{kk} = SM_est;
end
%%
Angle_save=[];
SM_est_save_all = [];
for kk = 1:1000 
    SM_est = SM_est_save{kk};
if isempty(SM_est)==0
    SM_est(SM_est(:,4)<500,:) = [];
    
    %
    saveAngle = [];
    saveOren = [];
    for ll = 1:size(SM_est,1)
        [mux,muy,muz,rotMobil] = secondM2SymmCone_RoSEO3D(double(SM_est(ll,:)),mean(mean(backg)),imgPara);
        if muz<=0
            mux = -mux;
            muy = -muy;
            muz = -muz;
        end
        saveOren(ll,:) = [mux,muy,muz,rotMobil];
        [thetaD, phiD, alphaD] = symmCone2angle(mux,muy,muz,rotMobil);
        saveAngle = [thetaD,phiD,alphaD,rotMobil,3*pi-sqrt(rotMobil*8*pi^2+pi^2)];

        Angle_save = [Angle_save;kk,saveAngle];
        SM_est_save_all = [SM_est_save_all;kk,double(SM_est(ll,:))];
    end
end


end
%%
r=30;
x = SM_est_save_all(:,2);
y = SM_est_save_all(:,3);
z = SM_est_save_all(:,4);
thetaD = Angle_save(:,2);
phiD = Angle_save(:,3);
%%
x =[x; SM_est_save_all(:,2)];
y = [y;SM_est_save_all(:,3)];
z = [z;SM_est_save_all(:,4)];
thetaD = [thetaD;Angle_save(:,2)];
phiD =[phiD; Angle_save(:,3)];

%%
r=30;
indx = abs(x)<700 & abs(y)<700 & abs(z)<350 & z>-100;
figure();
hold on
scatter3(x(indx),y(indx),z(indx),[],thetaD(indx),'filled'); axis image; colorbar;
plot3([x(indx)-r*cosd(thetaD(indx)).*cosd(phiD(indx)),x(indx),x(indx)+r*cosd(thetaD(indx)).*cosd(phiD(indx))].',...
     [y(indx)-r*cosd(thetaD(indx)).*sind(phiD(indx)),y(indx),y(indx)+r*cosd(thetaD(indx)).*sind(phiD(indx))].',...
     [z(indx)-r*sind(thetaD(indx)),z(indx),z(indx)+r*sind(thetaD(indx))].','k');
thetaD1 = acos((350-z)./350)/pi*180; phiD1 = atan2(y,x)/pi*180; 
plot3([x(indx)-r*cosd(thetaD1(indx)).*cosd(phiD1(indx)),x(indx),x(indx)+r*cosd(thetaD1(indx)).*cosd(phiD1(indx))].',...
     [y(indx)-r*cosd(thetaD1(indx)).*sind(phiD1(indx)),y(indx),y(indx)+r*cosd(thetaD1(indx)).*sind(phiD1(indx))].',...
     [z(indx)-r*sind(thetaD1(indx)),z(indx),z(indx)+r*sind(thetaD1(indx))].','r');
R=350;
% [X,Y,Z] = sphere;
% X(Z>0)=nan;Y(Z>0)=nan;Z(Z>0)=nan;
% X = X*R+20;Y = Y*R+30;Z=Z*R+R;
% surf(X,Y,Z,'FaceAlpha',0.5); %colormap('hot')
%%
vector_est = [cosd(thetaD(indx)).*cosd(phiD(indx)),cosd(thetaD(indx)).*sind(phiD(indx)),sind(thetaD(indx))];
vector_per = real([cosd(thetaD1(indx)).*cosd(phiD1(indx)),cosd(thetaD1(indx)).*sind(phiD1(indx)),sind(thetaD1(indx))]);
angleD = real(acos(sum((vector_est.*vector_per),2))./pi*180);
%angleD(angleD>90)=180-angleD(angleD>90);
histogram(angleD,20); title('rotated thetaD');
%%
figure();
subplot(1,3,1);
scatter(thetaD(indx),z(indx),'filled','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); ylabel('z(nm)'); xlabel('theta(\circ)');
subplot(1,3,2);
scatter(phiD(indx),x(indx),'filled','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); ylabel('x(nm)'); xlabel('phi(\circ)');
subplot(1,3,3);
scatter(phiD(indx),y(indx),'filled','MarkerFaceAlpha',.2,'MarkerEdgeAlpha',.2); ylabel('y(nm)'); xlabel('phi(\circ)');
%%
count = 0;
detactability = 0;
SM_est_SM1 = [];
SM_est_SM2 = [];
Angle_est_SM1 = [];
Angle_est_SM2 = [];
NLL_est = [];
SM_outlier=[];
Angle_outlier=[];
for kk = 1:200
   SM_est_cur =  SM_est_save{kk};
   Angle_est_cur = Angle_save{kk};
   if size(SM_est_cur,1)==2
       detactability = detactability+1;
       if norm(SM_est_cur(1,1:3)-SM_GT(1,1:3))<norm(SM_est_cur(2,1:3)-SM_GT(1,1:3))
           SM_est_SM1(detactability,:) = SM_est_cur(1,:);
           SM_est_SM2(detactability,:) = SM_est_cur(2,:);
           Angle_est_SM1(detactability,:) = Angle_est_cur(1,:);
           Angle_est_SM2(detactability,:) = Angle_est_cur(2,:);
           
       else
           SM_est_SM1(detactability,:) = SM_est_cur(2,:);
           SM_est_SM2(detactability,:) = SM_est_cur(1,:);
           Angle_est_SM1(detactability,:) = Angle_est_cur(2,:);
           Angle_est_SM2(detactability,:) = Angle_est_cur(1,:);
           %NLL_est(detactability) = NLL_save{kk};
           
       end
   else%if size(SM_est_cur,1)==1
       count = count+1;
       SM_outlier{count} = SM_est_cur;
       Angle_outlier{count} = Angle_est_cur;
       %NLL_outlier{count} = NLL_save{kk};
   end
   
end
%%
SM_est_SM1=[];
SM_est_SM2=[];
Angle_est_SM1=[];
Angle_est_SM2=[];
for kk =1:count
  
SM_est_SM1(kk,:) = SM_outlier{kk};
SM_est_SM2(kk,:) = SM_outlier{kk};

Angle_est_SM1(kk,:) = Angle_outlier{kk};
Angle_est_SM2(kk,:) = Angle_outlier{kk};

%NLL_est(kk) = NLL_outlier{kk};
end


%%
figure();
hist(NLL_est(NLL_est<1000),100); title('negative loglikelihood'); hold on; plot([100,100],[0,100],'r--','LineWidth',2);
%%
options = {'FaceColor','b',...
            'EdgeColor','none'};
%indx = (abs(SM_est_SM1(:,3)-0)<80&abs(SM_est_SM1(:,1))<50&abs(SM_est_SM1(:,2))<50);
%Angle_est_SM1_phi = Angle_est_SM1(:,2);
%Angle_est_SM1_phi(Angle_est_SM1_phi<0)=Angle_est_SM1_phi(Angle_est_SM1_phi<0)+180;
%indx = abs(Angle_est_SM1_phi-45)<40;
indx = 1:length(SM_est_SM1(:,1));
%indx = SM_est_SM1(:,3)<200&abs(SM_est_SM1(:,1))<50&abs(SM_est_SM1(:,2))<50;
height = 10;
plotSize = [3,2];
figure();
subplot(plotSize(1),plotSize(2),1);
histogram(SM_est_SM1(indx,1),100,options{:});
hold on; plot([SM_GT(1,1),SM_GT(1,1)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM1(indx,1))),' ','std=',num2str(std(SM_est_SM1(indx,1)))];
title({'x_{est} of SM1';test}); xlabel('x(nm)'); ylabel('count');

subplot(plotSize(1),plotSize(2),2);
histogram(SM_est_SM2(indx,1),100,options{:});
hold on; plot([SM_GT(2,1),SM_GT(2,1)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM2(indx,1))),' ','std=',num2str(std(SM_est_SM2(indx,1)))];
title({'x_{est} of SM2';test}); xlabel('x(nm)'); ylabel('count');

subplot(plotSize(1),plotSize(2),3);
histogram(SM_est_SM1(indx,2),100,options{:});
hold on; plot([SM_GT(1,2),SM_GT(1,2)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM1(indx,2))),' ','std=',num2str(std(SM_est_SM1(indx,2)))];
title({'y_{est} of SM1';test}); xlabel('y(nm)'); ylabel('count');

subplot(plotSize(1),plotSize(2),4);
histogram(SM_est_SM2(indx,2),100,options{:});
hold on; plot([SM_GT(2,2),SM_GT(2,2)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM2(indx,2))),' ','std=',num2str(std(SM_est_SM2(indx,2)))];
title({'y_{est} of SM2';test}); xlabel('y(nm)'); ylabel('count');

subplot(plotSize(1),plotSize(2),5);
histogram(SM_est_SM1(indx,3),100,options{:});
hold on; plot([SM_GT(1,3),SM_GT(1,3)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM1(indx,3))),' ','std=',num2str(std(SM_est_SM2(indx,3)))];
title({'z_{est} of SM1';test}); xlabel('z(nm)'); ylabel('count');

subplot(plotSize(1),plotSize(2),6);
histogram(SM_est_SM2(indx,3),100,options{:});
hold on; plot([SM_GT(2,3),SM_GT(2,3)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(SM_est_SM2(indx,3))),' ','std=',num2str(std(SM_est_SM2(indx,3)))];
title({'z_{est} of SM2';test}); xlabel('z(nm)'); ylabel('count');
%%

figure();
subplot(plotSize(1),plotSize(2),1);
histogram(Angle_est_SM1(indx,1),100,options{:});
hold on; plot([angle_GT(1,1),angle_GT(1,1)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM1(indx,1))),' ','std=',num2str(std(Angle_est_SM1(indx,1)))];
title({'\theta_{est} of SM1';test}); xlabel('\theta(\circ)'); ylabel('count');

subplot(plotSize(1),plotSize(2),2);
histogram(Angle_est_SM2(indx,1),100,options{:});
hold on; plot([angle_GT(2,1),angle_GT(2,1)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM2(indx,1))),' ','std=',num2str(std(Angle_est_SM2(indx,1)))];
title({'\theta_{est} of SM2';test}); xlabel('\theta(\circ)'); ylabel('count');

Angle_est_SM1_phi = Angle_est_SM1(indx,2);
Angle_est_SM1_phi(Angle_est_SM1_phi<0)=Angle_est_SM1_phi(Angle_est_SM1_phi<0)+180;
subplot(plotSize(1),plotSize(2),3);
histogram(Angle_est_SM1_phi,100,options{:});
hold on; plot([angle_GT(1,2),angle_GT(1,2)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM1_phi)),' ','std=',num2str(std(Angle_est_SM1_phi))];
title({'\phi_{est} of SM1';test}); xlabel('\phi(\circ)'); ylabel('count');

subplot(plotSize(1),plotSize(2),4);
histogram(Angle_est_SM2(indx,2),100,options{:});
hold on; plot([angle_GT(2,2),angle_GT(2,2)],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM2(indx,2))),' ','std=',num2str(std(Angle_est_SM2(indx,2)))];
title({'\phi_{est} of SM2';test}); xlabel('\phi(\circ)'); ylabel('count');

subplot(plotSize(1),plotSize(2),5);
histogram(Angle_est_SM1(indx,5),100,options{:});
hold on; plot([0,0],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM1(indx,5))),' ','std=',num2str(std(Angle_est_SM1(indx,5)))];
title({'\Omega_{est} of SM1';test}); xlabel('\gamma'); ylabel('count');

subplot(plotSize(1),plotSize(2),6);
histogram(Angle_est_SM2(indx,5),100,options{:});
hold on; plot([0,0],[0,height],'r--','LineWidth',2); 
test = ['mean=',num2str(mean(Angle_est_SM2(indx,5))),' ','std=',num2str(std(Angle_est_SM2(indx,5)))];
title({'\Omega_{est} of SM2';test}); xlabel('\gamma'); ylabel('count');


