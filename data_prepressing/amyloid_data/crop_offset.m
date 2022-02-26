
fileFolder = 'E:\Experimental_data\20220214 a-beta amyloid\';
offsetName = 'processes data\20220214_offSet_for_amyloid.mat';
load(strcat(fileFolder,'processes data\saved_beads_loc_for_tform\tformx2y_y_center_469_198_FoV_350_for data103_149.mat'));
ROI_centerY = [469,198]; 
ROI_centerX = transformPointsInverse(tformx2y,[1024,0]+[-ROI_centerY(1),ROI_centerY(2)])+[1024,0];

Nimg = 1;
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
SMLM_save_Nmae = ['processes data\offset_centerY_y',num2str(ROI_centerY(1)),'_x_',num2str(ROI_centerY(2)),'_','FoV',num2str(FoV(1)),'_',num2str(FoV(2)),'_',num2str(count),'th_FoV','.mat'];

ROI_centerY_cur = round(ROI_centerY+[center_x(ii),center_y(jj)]);
ROI_centerX_cur = round(transformPointsInverse(tformx2y,[1024,0]+[-ROI_centerY_cur(1),ROI_centerY_cur(2)])+[1024,0]);


load([fileFolder,offsetName]);
SMLM_img = offset;
SMLM_img_ROIy = double(SMLM_img(ROI_centerY_cur(2)+range,ROI_centerY_cur(1)+range));
SMLM_img_ROIx = double(SMLM_img(ROI_centerX_cur(2)+range,ROI_centerX_cur(1)+range));
SMLM_img = [SMLM_img_ROIx,fliplr(SMLM_img_ROIy)];
offset = SMLM_img;
save([fileFolder,SMLM_save_Nmae],'offset');

end
end

%%
