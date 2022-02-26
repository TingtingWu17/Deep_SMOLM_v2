
dataN =164;
fileFolder = 'E:\Experimental_data\20220214 a-beta amyloid\';
SMLMName = ['_',num2str(dataN),'\_',num2str(dataN),'_MMStack_Default.ome.tif'];


Nimg = 10;

%
ROI_centery = [187,586]; 
ROI_centerx = [161,1476]; 
D = 41;
R = (D-1)/2;
load([fileFolder,'processes data\20220214_offSet_for_amyloid.mat']);
offset_ROIy = double(offset(ROI_centery(1)-R:ROI_centery(1)+R,ROI_centery(2)-R:ROI_centery(2)+R));
offset_ROIx = double(offset(ROI_centerx(1)-R:ROI_centerx(1)+R,ROI_centerx(2)-R:ROI_centerx(2)+R));
%

SMLM_imgR = Tiff([fileFolder,SMLMName],'r');
for i=1:Nimg

    setDirectory(SMLM_imgR,i);
    SMLM_img = double(SMLM_imgR.read);
    SMLM_img_ROIy = double(SMLM_img(ROI_centery(1)-R:ROI_centery(1)+R,ROI_centery(2)-R:ROI_centery(2)+R))-offset_ROIy;
    SMLM_img_ROIx = double(SMLM_img(ROI_centerx(1)-R:ROI_centerx(1)+R,ROI_centerx(2)-R:ROI_centerx(2)+R))-offset_ROIx;
    SMLM_img_save(:,:,i) = [SMLM_img_ROIx,SMLM_img_ROIy];
    
end
save([fileFolder,'processes data\beads_for_phase_retrival\data',num2str(dataN),'_beads1_L',num2str(ROI_centery(1)),'_',num2str(ROI_centery(2)),...
    '_R_',num2str(ROI_centerx(1)),'_',num2str(ROI_centerx(2)),...
     '_wo_offset_unfliped.mat'],"SMLM_img_save")
%%

