%% descreption: combine multiply mat file into a larger mat file
%%
clear;
clc;
%% 
for channel=[0:4,8]
folder = 'simulate_image_May5_SNR380vs2_flipped/';
folder2 = 'trainingDataMay5_SNR380vs2_flipped/';
image_size = 80;  % the pixel size of the simulation image (feel free to change it)
upsampling_ratio  = 6;
pmask = 'clear plane.bmp';
%% user defined parameters
n_images = 1000; % the simulated image numbers (feel free to change it)
% signal=380; %(feel free to change it)
background=2; %(feel free to change it)
density = 9; %the maximum number of molecules in an image %(feel free to change it)
range_density = 4;
min_density = 7;
img_pixels = 240;
group_num = 20;
num_train = group_num*n_images;

% density_density = zeros(1,num_train);
switch channel
    case 0
        image_intensity = zeros(img_pixels,img_pixels,num_train);       
    case 1
        image_raw_xy = zeros(2,img_pixels,img_pixels,num_train);
    case 2
        image_theta = zeros(img_pixels,img_pixels,num_train);
    case 3
        image_phi = zeros(img_pixels,img_pixels,num_train);
    case 4
        image_gamma = zeros(img_pixels,img_pixels,num_train);
    case 5
        image_raw_xy_denoised = zeros(2,img_pixels,img_pixels,num_train);
    case 6
        image_dx = zeros(img_pixels,img_pixels,num_train);
    case 7
        image_dy = zeros(img_pixels,img_pixels,num_train); 
    case 8
        image_omega = zeros(img_pixels,img_pixels,num_train);
    
end
for ii = 1:group_num %each 1000 images, and total 1000*10 images
    switch channel
        case 0
            load([folder 'image_intensity_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_intensity(:,:,(ii-1)*n_images+1:ii*n_images) = image_intensity_save;       
        case 1
            load([folder 'image_raw_x_density',num2str(density),'_group',num2str(ii),'.mat']);
            load([folder 'image_raw_y_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_raw_xy(1,:,:,(ii-1)*n_images+1:ii*n_images) = image_raw_x_save;
            image_raw_xy(2,:,:,(ii-1)*n_images+1:ii*n_images) = image_raw_y_save;
        case 2
            load([folder 'image_theta_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_theta(:,:,(ii-1)*n_images+1:ii*n_images) = image_theta_save;
        case 3
            load([folder 'image_phi_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_phi(:,:,(ii-1)*n_images+1:ii*n_images) = image_phi_save;
        case 4
            load([folder 'image_gamma_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_gamma(:,:,(ii-1)*n_images+1:ii*n_images) = image_gamma_save;
        case 5
            load([folder 'imagex_without_Poisson',num2str(density),'_group',num2str(ii),'.mat']);
            load([folder 'imagey_without_Poisson',num2str(density),'_group',num2str(ii),'.mat']);
            image_raw_xy_denoised(1,:,:,(ii-1)*n_images+1:ii*n_images) = image_raw_x_denoised_save;%image_raw_x_save;
            image_raw_xy_denoised(2,:,:,(ii-1)*n_images+1:ii*n_images) = image_raw_y_denoised_save;%image_raw_y_save;
       case 6
            load([folder 'image_dx_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_dx(:,:,(ii-1)*n_images+1:ii*n_images) = image_dx_save;
        case 7
            load([folder 'image_dy_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_dy(:,:,(ii-1)*n_images+1:ii*n_images) = image_dy_save;
        case 8
            load([folder 'image_omega_density',num2str(density),'_group',num2str(ii),'.mat']);
            image_omega(:,:,(ii-1)*n_images+1:ii*n_images) = image_omega_save;
    end
end
switch channel
    case 0
        save([folder2,'TrainingSet_v4_image_intensity_',num2str(group_num),'G.mat'],'image_intensity','-v7.3');
    case 1
        save([folder2,'TrainingSet_v4_image_raw_xy_',num2str(group_num),'G.mat'],'image_raw_xy','-v7.3');
    case 2
        save([folder2,'TrainingSet_v4_image_theta_',num2str(group_num),'G.mat'],'image_theta','-v7.3');
    case 3
        save([folder2,'TrainingSet_v4_image_phi_',num2str(group_num),'G.mat'],'image_phi','-v7.3');
    case 4
        save([folder2,'TrainingSet_v4_image_gamma_',num2str(group_num),'G.mat'],'image_gamma','-v7.3');
    case 5
        save([folder2,'TrainingSet_v4_image_raw_xy_denoised_',num2str(group_num),'G.mat'],'image_raw_xy_denoised','-v7.3');
    case 6
        save([folder2,'TrainingSet_v4_image_dx_',num2str(group_num),'G.mat'],'image_dx','-v7.3');
    case 7
        save([folder2,'TrainingSet_v4_image_dy_',num2str(group_num),'G.mat'],'image_dy','-v7.3');
    case 8
        save([folder2,'TrainingSet_v4_image_omega_',num2str(group_num),'G.mat'],'image_omega','-v7.3');
   
end
clear;
clc;
end