
for dataN =[119:123]
fileFolder = 'E:\Experimental_data\20220214 a-beta amyloid\';
SMLMName = ['_',num2str(dataN),'\_',num2str(dataN),'_MMStack_Default.ome.tif'];
load(strcat(fileFolder,'processes data\saved_beads_loc_for_tform\tformx2y_y_center_469_198_FoV_350_for data103_149.mat'));
ROI_centerY = [469,198]; 
ROI_centerX = transformPointsInverse(tformx2y,[1024,0]+[-ROI_centerY(1),ROI_centerY(2)])+[1024,0];

Nimg = 2000;
FoV = [320,320]; 
N_FoV = [4,4];
FoV_each = 96;

center_x = FoV(1)/N_FoV(1)/2*[-N_FoV(1)+1:2:N_FoV(1)-1];
center_y = FoV(2)/N_FoV(2)/2*[-N_FoV(2)+1:2:N_FoV(2)-1];
count = 0;


for ii = 1:length(center_x)
for jj = 1:length(center_y)

count = count+1;
range = round(-FoV_each/2+1):1:round(FoV_each/2);
SMLM_save_Nmae = ['processes data\data',num2str(dataN),'_centerY_y',num2str(ROI_centerY(1)),'_x_',num2str(ROI_centerY(2)),'_','FoV',num2str(FoV(1)),'_',num2str(FoV(2)),'_',num2str(count),'th_FoV','.tif'];

ROI_centerY_cur = round(ROI_centerY+[center_x(ii),center_y(jj)]);
ROI_centerX_cur = round(transformPointsInverse(tformx2y,[1024,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[1024,0]);
SMLM_imgR = Tiff([fileFolder,SMLMName],'r');


for i=1:Nimg

    setDirectory(SMLM_imgR,i);
    SMLM_img = double(SMLM_imgR.read);
    SMLM_img_ROIy = uint16(SMLM_img(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range));
    SMLM_img_ROIx = uint16(SMLM_img(ROI_centerX_cur(2)+range,ROI_centerX_cur(1)+range));
    SMLM_img = [SMLM_img_ROIx,fliplr(SMLM_img_ROIy)];
    

    if i==1
     imwrite(SMLM_img,[fileFolder,SMLM_save_Nmae])
    else
    imwrite(SMLM_img,[fileFolder,SMLM_save_Nmae],'WriteMode','append')
    end
end
end
end
end
%%
