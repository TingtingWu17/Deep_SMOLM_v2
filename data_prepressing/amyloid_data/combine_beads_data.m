dataN = [152:3:164];

fileFolder = 'E:\Experimental_data\20220214 a-beta amyloid\processes data\beads_for_phase_retrival\';
beads_img = [];
for ii = dataN

mainDirContents = dir(fileFolder);
mask = startsWith({mainDirContents.name},['data',num2str(ii)]);
for jj = 1:length(mask)
    if mask(jj)==1
        load([fileFolder,mainDirContents(jj).name])
        beads_img = cat(3,beads_img,SMLM_img_save);
    end
end
end

save([fileFolder,'combine_beads_data_',num2str(dataN(1)),'_to_data_',num2str(dataN(end)),'.mat'],"beads_img");