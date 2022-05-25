save_folder = '/home/wut/Documents/Deep-SMOLM/data/opt_PSF_data_1000vs2/training_20220216_retrieve_pixOL_com_SNR1000vs4_large_variance_corrected_angle_uniform_distribution_little_3D/'; 


for ii = 1:90000
    
    load([save_folder,'image_with_poission_bkgdRmvd_up',num2str(ii),'.mat']);
    load([save_folder,'image_with_poission',num2str(ii),'.mat'],'image_with_poission');
    N_pixel = size(image_with_poission,2)^2*2;
    imSz = size(image_with_poission,2);
    bkg_x = (sum(image_with_poission,'all') - sum(image_with_poission_bkgdRmvd_up,'all')/6)/N_pixel;
    bkg_y = bkg_x;
    img_bkg_x = reshape(ones(imSz,imSz)*bkg_x,1,imSz,imSz);
    img_bkg_y = reshape(ones(imSz,imSz)*bkg_y,1,imSz,imSz);
    img_bkg = cat(1,img_bkg_x,img_bkg_y);
    save([save_folder,'img_bkg',num2str(ii),'.mat'],'img_bkg');
    delete([save_folder,'img_bkg_y',num2str(ii),'.mat']);
    delete([save_folder,'img_bkg_x',num2str(ii),'.mat']);
    
end