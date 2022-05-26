%% plot reconstruction video
%Image = Tiff('C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\figure_generation_data\Fig3\reconstructed_filted\reconstruction-1.tif','r');
Image = Tiff('C:\Users\wutt0\OneDrive - Washington University in St. Louis\github\data_of_PSF_optimization\figure_generation_data\Fig3\reconstructed_scaled_filtered\reconstruction_scaled.tif','r');
for ii = 1:99
setDirectory(Image,ii)
image_cur(:,:,:,ii) = Image.read;
end

%
v = VideoWriter('Movie1_reconstruction.mp4','MPEG-4');
v.Quality = 95;
v.FrameRate = 2;

open(v);
%v.VideoCompressionMethod = 'H.264';
for ii = 1:99
    A = image_cur(:,:,:,ii);
    writeVideo(v,A);
end
close(v);



%% plot reconstruction video
fileFolder1 = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data_of_PSF_optimization\figure_generation_data\Fig3\spikes-plane\';
fileFolder2 = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data_of_PSF_optimization\figure_generation_data\Fig3\spikes-plane_DPPC\';


%
v = VideoWriter('h_sections.mp4','MPEG-4');
v.Quality = 95;
v.FrameRate = 7;

open(v);
%v.VideoCompressionMethod = 'H.264';
for ii = 3:50
    A = imread([fileFolder1,num2str(ii),'.png']);
    B = imread([fileFolder2,num2str(ii),'.png']);
    C = cat(1,A,B);
    writeVideo(v,C);
end
close(v);

%% plot reconstruction video
fileFolder1 = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\figure_generation_data\Fig3\spikes\';
fileFolder2 = 'C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\data of PSF-optimization\figure_generation_data\Fig3\spikes_DPPC\';


%
v = VideoWriter('3D_sphere.mp4','MPEG-4');
%v.VideoCompressionMethod = 'H.264';
v.Quality = 95;
v.FrameRate = 7;

open(v);
%v.VideoCompressionMethod = 'H.264';
for ii = 1:50
    A = imread([fileFolder1,num2str(ii),'.png']);
    B = imread([fileFolder2,num2str(ii),'.png']);
    C = cat(1,A,B);
    writeVideo(v,C);
end

close(v);

