%% 2020/02/20 Ting: correct the noise model; 
%                   generate a small size basis matrix
%                   change to high SNR situation: signal 1000, background 5
%% 2020/02/22 Asheq: Modified saving the x and y channel psf map without
% noise

%%

% clear;
% clc;
% %%
% save_folder = 'simulate_image_correct_noise_Feb22\'; 
% % generate a small size basis and upsampling it
% 
% image_size = 17;  % the pixel size of the simulation image (feel free to change it)
% upsampling_ratio  = 6;
% pmask = 'clear plane.bmp';
% basis_matrix_SD_s1 = forward_model(pmask, image_size);
% imsize = size(basis_matrix_SD_s1);
% basis_matrix_SD_temp = reshape(basis_matrix_SD_s1,image_size,image_size*2,6);
% %
% for i = 1:6
%   basis_matrix_SD(:,:,i) = imresize(basis_matrix_SD_temp(:,:,i),[image_size,image_size*2]*upsampling_ratio,'box');
% end
% 
% save([save_folder,'basis_matrix_SD.mat'],'basis_matrix_SD');



%% image generation
clear;
clc;

% give the save address for generated data
% ********************************
save_folder = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data for Deep-SMOLM\validation_20210331_2SM_fixed\'; 
% ********************************
image_size = 56;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'pixOL_v12.bmp';
pixel_size_xy = 58.6; %in unit of nm
pixel_size_z = 50; % in unit of nm

distance_differ_set = [1,2,3,4,5,6,7,8,9,10]; %in unit of pixel
%% user defined parameters
%% gaussian filter
h_shape = [7,7];
h_sigma = 1;
[x,y] = meshgrid([-(h_shape(1)-1)/2:(h_shape(1)-1)/2]);
h = exp(-(x.^2+y.^2)/(2*h_sigma^2));
h = h./max(max(h));
%% user defined parameters

n_images = 1; % the simulated image numbers (feel free to change it)
signal= 1000; %(feel free to change it)
background=2 ; %(feel free to change it)
%signal_sigma = 2000;
NFP = -500; %(nm); NFP: Normal forcal plane
z_range_phy = [0,1000]; %(nm); the axial location range of SMs
load('imgPara');
%##############run thes two lines only if you change the parameters#############
%imgPara = forward_model_opt_3D(pmask, image_size,NFP,z_range_phy,pixel_size_xy,pixel_size_z);
%save('imgPara.mat','imgPara');
%###########################################################################
imgPara.img_sizex = image_size;
imgPara.img_sizey = image_size;
f_forwardModel = @(x,G_re,b) abs((G_re*x)+b);

kk=0;
for ii = 1:200*length(distance_differ_set)  %each 4 images, and total 2000*4 images
if rem(ii-1,200)==0
   ii
   kk=kk+1;
   kk
end


image_with_poission = zeros(2,image_size,image_size);
image_with_poission_up = zeros(2,image_size*upsampling_ratio,image_size*upsampling_ratio);
image_GT_up = zeros(5,image_size*upsampling_ratio,image_size*upsampling_ratio);

n_SMs = 2; % number of single molecules
[thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs);
%theta angle of SMs, note theta is in the range of (0,90) degree
%phi angle of SMs, note phi is in the range of (0,360) degree
%gamma (orientaiton constraint) is used to represent alpha angle. it is in the range of (0,1)


x_SMs1 = (0.9999*rand(1)-1/2)/upsampling_ratio; %x location, in unit of pixles
y_SMs1 = (0.9999*rand(1)-1/2)/upsampling_ratio;%y location, in unit of pixles
z_SMs = (rand(1,n_SMs)*(z_range_phy(2)-z_range_phy(1))+z_range_phy(1))./pixel_size_z; %z location, in unit of pixels
angle = rand(1)*360;
x_dif = distance_differ_set(kk)*cosd(angle);
y_dif = distance_differ_set(kk)*sind(angle);
x_SMs2 = x_SMs1+x_dif;
y_SMs2 = y_SMs1+y_dif;

x_SMs = [x_SMs1,x_SMs2];
y_SMs = [y_SMs1,y_SMs2];

temp = (poissrnd(3,1,100)+normrnd(0,1,1,100)-0.5)*350; temp(temp<100)=[];
signal_SMs = temp(1:n_SMs);
x_SMs_phy = x_SMs*pixel_size_xy;
y_SMs_phy = y_SMs*pixel_size_xy;
z_SMs_phy = z_SMs*pixel_size_z;

% save the list of the ground truth
x_grd= x_SMs.'; y_grd= y_SMs.';  z_grd= z_SMs.'; x_phy= x_SMs_phy.'; y_phy= y_SMs_phy.';  z_phy= z_SMs_phy.'; 
thetaD_grd= thetaD_SMs.'; phiD_grd=phiD_SMs.'; 
gamma_grd = gamma_SMs.'; I_grd = signal_SMs.'; 




%% forward imaging system

[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M(thetaD_SMs,phiD_SMs,gamma_SMs);
M = [muxx;muyy;muzz;muxy;muxz;muyz];
[lambda,loc] = generate_lambda(signal_SMs,x_SMs_phy,y_SMs_phy,z_SMs_phy,M,imgPara);
[G_re,loc_re_new,lambda] = update_basisMatrix(n_SMs,lambda,loc,imgPara);
I_SMs = f_forwardModel(lambda,G_re,background);
I_SMs = reshape(I_SMs,image_size,image_size*2);
I_SMsx = I_SMs(1:image_size,1:image_size,:);
I_SMsy = I_SMs(1:image_size,image_size+1:image_size*2,:);
I_SMsy = flip(I_SMsy,2);

%% generate the basis image
I = ones(image_size,image_size*2)*background;
Ix = I(1:image_size,1:image_size);
Iy = I(1:image_size,image_size+1:image_size*2);
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
I_sXX = I_intensity_up;
I_sYY = I_intensity_up;
I_sZZ = I_intensity_up;
I_sXY = I_intensity_up;
I_sXZ = I_intensity_up;
I_sYZ = I_intensity_up;

h_basis = I_basis;
h_basis(round((size(I_basis,1)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2],round((size(I_basis,2)+1)/2)+[-(h_shape(1)-1)/2:(h_shape(1)-1)/2]) = h;
compensate = upsampling_ratio/2+1; %register the coordinate for even images;
I_basis(round((size(I_basis,1)+1)/2)-compensate,round((size(I_basis,2)+1)/2)-compensate) = 1;

I = I_SMs;
Ix = I_SMsx;
Iy = I_SMsy;
for i = 1:n_SMs

temp1 = imtranslate(I_basis,[round(x_SMs(i)*upsampling_ratio),round(y_SMs(i)*upsampling_ratio)]);
I_intensity_up = I_intensity_up+temp1*signal_SMs(i);
I_theta_up = I_theta_up+temp1*thetaD_SMs(i);
I_phi_up = I_phi_up+temp1*phiD_SMs(i);
I_gamma_up = I_gamma_up+temp1*gamma_SMs(i);
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
end

I_poissx = poissrnd(Ix); % if you need multiple realization for a single ground truth, modify here
%imagesc(I_poiss); axis image;
I_poissy = poissrnd(Iy);
I_poissx_up = imresize(I_poissx,[image_size,image_size]*upsampling_ratio,'box');  
I_poissy_up = imresize(I_poissy,[image_size,image_size]*upsampling_ratio,'box'); 
Ix_up = imresize(Ix,[image_size,image_size]*upsampling_ratio,'box');  
Iy_up = imresize(Iy,[image_size,image_size]*upsampling_ratio,'box'); 

%save ground truth and image
image_with_poission(1,:,:) = I_poissx;
image_with_poission(2,:,:) = I_poissy;
image_with_poission_up(1,:,:) = I_poissx_up;
image_with_poission_up(2,:,:) = I_poissy_up;
image_noiseless(1,:,:) = Ix;
image_noiseless(2,:,:) = Iy;
image_noiseless_up(1,:,:) = Ix_up;
image_noiseless_up(2,:,:) = Iy_up;
image_GT_up(1,:,:) = I_intensity_up;
image_GT_up(2,:,:) = I_theta_up;
image_GT_up(3,:,:) = I_phi_up;
image_GT_up(4,:,:) = I_gamma_up;
image_GT_up(5,:,:) = I_intensity_gaussian;
image_GT_up(6,:,:) = I_sXX;
image_GT_up(7,:,:) = I_sYY;
image_GT_up(8,:,:) = I_sZZ;
image_GT_up(9,:,:) = I_sXY;
image_GT_up(10,:,:) = I_sXZ;
image_GT_up(11,:,:) = I_sYZ;
GT_list(1,:)=x_phy;
GT_list(2,:)=y_phy;
GT_list(3,:)=z_phy;
GT_list(4,:)=I_grd;
GT_list(5,:)=thetaD_grd;
GT_list(6,:)=phiD_grd;
GT_list(7,:)=gamma_grd;

image_with_poission_bkgdRmvd = image_with_poission-background;
image_with_poission_bkgdRmvd_up = image_with_poission_up-background;


save([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
save([save_folder,'image_with_poission_up',num2str(ii),'.mat'],'image_with_poission_up');
save([save_folder,'image_with_poission_bkgdRmvd',num2str(ii),'.mat'],'image_with_poission_bkgdRmvd');
save([save_folder,'image_with_poission_bkgdRmvd_up',num2str(ii),'.mat'],'image_with_poission_bkgdRmvd_up');
save([save_folder,'image_GT_up',num2str(ii),'.mat'],'image_GT_up');
save([save_folder,'GT_list',num2str(ii),'.mat'],'GT_list');
save([save_folder,'image_noiseless',num2str(ii),'.mat'],'image_noiseless');
save([save_folder,'image_noiseless_up',num2str(ii),'.mat'],'image_noiseless_up');

end