%% description: for generating training data of trispot PSF
% the 4 channel GT data + poisson data are *combined* together for quick
% reading

%% 2020/02/20 Ting: correct the noise model; 
%                   generate a small size basis matrix
%                   change to high SNR situation: signal 1000, background 5
% 2020/02/22 Asheq: Modified saving the x and y channel psf map without
% noise

clear;
clc;


%% parameter of the microscopy

% give the save address for generated data
% ********************************

save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/validation_20220711_SNR1000vs2_1SM_0909upsampled_gradient_model/'; 
% ********************************
image_size = 32;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12';
%basis_matrix_opt = forward_model_opt(pmask, image_size);
%pmask_retrieve_name = '20220528_pixOLcom_retrieved.mat';


%
pixel_size_xy = 58.6; %in unit of nm
pixel_size_z = 20; % in unit of nm

NFP = 0; %(nm); NFP: Normal forcal plane
z_range_phy = [-150,150]; %(nm); the axial location range of SMs
%load('imgPara');
%##############run thes two lines only if you change the parameters#############
%imgPara = forward_model_opt_3D_retrieved(pmask, image_size,NFP,z_range_phy,pixel_size_xy,pixel_size_z,pmask_retrieve_name);
imgPara = forward_model_opt_3D(pmask, image_size,NFP,z_range_phy,pixel_size_xy,pixel_size_z);
%save('imgPara.mat','imgPara');
%###########################################################################
imgPara.img_sizex = image_size;
imgPara.img_sizey = image_size;
f_forwardModel = @(x,G_re) abs((G_re*x));

%% gaussian filter
h_shape = [7,7];
h_sigma = 1;
[x,y] = meshgrid([-(h_shape(1)-1)/2:(h_shape(1)-1)/2]);
h = exp(-(x.^2+y.^2)/(2*h_sigma^2));
h = h./max(max(h));


%% user defined parameters

n_images = 1; % the simulated image numbers (feel free to change it)
signal= 1000; %(feel free to change it)
background_avg=2; %(feel free to change it)
%signal_sigma = 2000;
SM_num_range = 8;
SM_num_min = 7;


for ii = 1:2000%each 4 images, and total 2000*4 images
if rem(ii,100)==0
   ii
end
x_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation
y_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation
x_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the xlocation (phyiscal distance)
y_phy = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the ylocation (phyiscal distance)
thetaD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the thetaD
phiD_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the phi
gamma_grd = nan(SM_num_range+SM_num_min,1); %for saving the groundtruth of the gamma 
I_grd = nan(SM_num_range+SM_num_min,1);

image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 1; %floor(rand(1)*SM_num_range+SM_num_min); % number of single molecules
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs);
%[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_gamma_linear_distribution(n_SMs);
%[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD_with_M_uniformly_sampled_v2(n_SMs);

%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)


x_SMs = 0.9/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio; %x location, in unit of pixles
y_SMs = 0.9/upsampling_ratio;%(0.9999*rand(1)-1/2)/upsampling_ratio;%y location, in unit of pixles
%z_SMs = (rand(1,n_SMs)*300-150)/pixel_size_z;
z_SMs = 0;
%temp=generateSignal_distribution(); temp(temp<100)=[];
%temp = (poissrnd(3,1,100000)+normrnd(0,1,1,100000)-0.5)*350; temp(temp<100)=[]; hist(temp,1000); mean(temp)
temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<80)=[];
temp = temp;
signal_SMs = 1000;
x_SMs_phy = x_SMs*pixel_size_xy;
y_SMs_phy = y_SMs*pixel_size_xy;
z_SMs_phy = z_SMs*pixel_size_z;

% save the list of the ground truth
x_grd(1:n_SMs) = x_SMs.'; y_grd(1:n_SMs) = y_SMs.';  x_phy(1:n_SMs) = x_SMs_phy.'; y_phy(1:n_SMs) = y_SMs_phy.'; 
thetaD_grd(1:n_SMs) = thetaD_SMs.'; phiD_grd(1:n_SMs)=phiD_SMs.'; 
gamma_grd(1:n_SMs) = gamma_SMs.'; I_grd(1:n_SMs) = signal_SMs.'; 

background = background_avg;
bkg_img = [ones(image_size,image_size)*background,ones(image_size,image_size)*background];
bkg_img_up1 = imresize(bkg_img(:,1:image_size),[image_size,image_size]*upsampling_ratio,'box');  
bkg_img_up2 = imresize(bkg_img(:,image_size+1:image_size*2),[image_size,image_size]*upsampling_ratio,'box');
bkg_img_up(1,:,:)=bkg_img_up1;
bkg_img_up(2,:,:)=bkg_img_up2;


%% forward imaging system


[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];

[lambda,loc] = generate_lambda(signal_SMs,x_SMs_phy,y_SMs_phy,z_SMs_phy,M,imgPara);
[G_re,loc_re_new,lambda] = update_basisMatrix(n_SMs,lambda,loc,imgPara);
I_SMs = f_forwardModel(lambda,G_re);
I_SMs = reshape(I_SMs,image_size,image_size*2);
I_SMs = I_SMs+bkg_img;
Ix = I_SMs(1:image_size,1:image_size,:);
Iy = I_SMs(1:image_size,image_size+1:image_size*2,:);

%% generate the basis image

% I = imresize(I,size(I)*upsampling_ratio,'nearest');
I_basis = zeros(image_size*upsampling_ratio,image_size*upsampling_ratio);
%I_basis = imresize(I_basis,size(I_basis)*upsampling_ratio,'nearest');

% four channels
I_intensity_up = I_basis;
I_theta_up = I_intensity_up;
I_phi_up = I_intensity_up;
I_omega_up = I_intensity_up;
I_gamma_up = I_intensity_up;
I_dx_up = I_intensity_up;
I_dy_up = I_intensity_up;
I_intensity_gaussian = I_intensity_up;
I_XX = I_intensity_up;
I_YY = I_intensity_up;
I_ZZ = I_intensity_up;
I_XY = I_intensity_up;
I_XZ = I_intensity_up;
I_YZ = I_intensity_up;
I_sXX = I_intensity_up;
I_sYY = I_intensity_up;
I_sZZ = I_intensity_up;
I_sXY = I_intensity_up;
I_sXZ = I_intensity_up;
I_sYZ = I_intensity_up;
I_gamma_heatmap = I_intensity_up;
I_sXX_wogamma = I_intensity_up;
I_sYY_wogamma = I_intensity_up;
I_sZZ_wogamma = I_intensity_up;

h_basis = I_basis;
h_basis(round((size(I_basis,1)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2],round((size(I_basis,2)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2]) = h;
I_basis(round((size(I_basis,1)+1)/2),round((size(I_basis,2)+1)/2)) = 1;


for i = 1:n_SMs

temp1 = imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)]);
I_intensity_up = I_intensity_up+temp1*signal_SMs(i);
I_theta_up = I_theta_up+temp1*thetaD_SMs(i);
I_phi_up = I_phi_up+temp1*phiD_SMs(i);

%I_dx_up = I_dx_up+temp1*(x_SMs(i)*upsampling_ratio-round(x_SMs(i)*upsampling_ratio));
%I_dy_up = I_dy_up+temp1*(y_SMs(i)*upsampling_ratio-round(y_SMs(i)*upsampling_ratio));

temp = imtranslate(h_basis,[(x_SMs(i)*upsampling_ratio),(y_SMs(i)*upsampling_ratio)],'bicubic');
I_intensity_gaussian = I_intensity_gaussian+temp*signal_SMs(i);

I_sXX = I_sXX+temp*signal_SMs(i)*muxx(i);
I_sYY = I_sYY+temp*signal_SMs(i)*muyy(i);
I_sZZ = I_sZZ+temp*signal_SMs(i)*muzz(i);
I_sXY = I_sXY+temp*signal_SMs(i)*muxy(i);
I_sXZ = I_sXZ+temp*signal_SMs(i)*muxz(i);
I_sYZ = I_sYZ+temp*signal_SMs(i)*muyz(i);

I_XX = I_XX+temp*muxx(i);
I_YY = I_YY+temp*muyy(i);
I_ZZ = I_ZZ+temp*muzz(i);
I_XY = I_XY+temp*muxy(i);
I_XZ = I_XZ+temp*muxz(i);
I_YZ = I_YZ+temp*muyz(i);

end

I_poissx = poissrnd(Ix); % if you need multiple realization for a single ground truth, modify here
% %imagesc(I_poiss); axis image;
I_poissy = poissrnd(Iy);
% I_poissx_up = imresize(I_poissx,[image_size,image_size]*upsampling_ratio,'box');  
% I_poissy_up = imresize(I_poissy,[image_size,image_size]*upsampling_ratio,'box'); 
% Ix_up = imresize(Ix,[image_size,image_size]*upsampling_ratio,'box');  
% Iy_up = imresize(Iy,[image_size,image_size]*upsampling_ratio,'box'); 

%save ground truth and image
image_with_poission(1,:,:) = I_poissx;
image_with_poission(2,:,:) = I_poissy;
% image_with_poission_up(1,:,:) = I_poissx_up;
% image_with_poission_up(2,:,:) = I_poissy_up;
image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;
% image_noiseless_up(1,:,:) = Ix_up;
% image_noiseless_up(2,:,:) = Iy_up;
% image_GT_up(1,:,:) = I_intensity_up;
% image_GT_up(2,:,:) = I_theta_up;
% image_GT_up(3,:,:) = I_phi_up;
% image_GT_up(4,:,:) = I_gamma_up;
image_GT_up(1,:,:) = I_intensity_gaussian;
image_GT_up(2,:,:) = I_XX;
image_GT_up(3,:,:) = I_YY;
image_GT_up(4,:,:) = I_ZZ;
image_GT_up(5,:,:) = I_XY;
image_GT_up(6,:,:) = I_XZ;
image_GT_up(7,:,:) = I_YZ;

image_GT_up(8,:,:) = I_sXX;
image_GT_up(9,:,:) = I_sYY;
image_GT_up(10,:,:) = I_sZZ;
image_GT_up(11,:,:) = I_sXY;
image_GT_up(12,:,:) = I_sXZ;
image_GT_up(13,:,:) = I_sYZ;


% image_GT_up(12,:,:) = I_gamma_heatmap;
% image_GT_up(13,:,:) = I_sXX_wogamma;
% image_GT_up(14,:,:) = I_sYY_wogamma;
% image_GT_up(15,:,:) = I_sZZ_wogamma;

GT_list(1,:)=ones(size(x_phy))*ii;
GT_list(2,:)=x_phy;
GT_list(3,:)=y_phy;
GT_list(4,:)=I_grd;
GT_list(5,:)=thetaD_grd;
GT_list(6,:)=phiD_grd;
GT_list(7,:)=gamma_grd;
img_bkg(1,:,:) = bkg_img(1:image_size,1:image_size);
img_bkg(2,:,:) = bkg_img(1:image_size,(1:image_size)+image_size);
% image_with_poission_bkgdRmvd = image_with_poission-background;
% image_with_poission_bkgdRmvd_up = image_with_poission_up-background;


save([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
save([save_folder,'img_bkg',num2str(ii),'.mat'],'img_bkg');
%save([save_folder,'image_with_poission_up',num2str(ii),'.mat'],'image_with_poission_up');
%save([save_folder,'image_with_poission_bkgdRmvd',num2str(ii),'.mat'],'image_with_poission_bkgdRmvd');
%save([save_folder,'image_with_poission_bkgdRmvd_up',num2str(ii),'.mat'],'image_with_poission_bkgdRmvd_up');
save([save_folder,'image_GT_up',num2str(ii),'.mat'],'image_GT_up');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');
%save([save_folder,'image_noiseless_up',num2str(ii),'.mat'],'image_noiseless_up');




end