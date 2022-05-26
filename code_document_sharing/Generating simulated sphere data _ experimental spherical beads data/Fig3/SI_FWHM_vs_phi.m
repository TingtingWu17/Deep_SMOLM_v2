

%Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,6.84]);
plot_FWHMvs_phi();


%% functions


function plot_FWHMvs_phi()

for kk = 1:2
    if kk==1
        load('est_xyCentered_v28.mat')
        load('data_for_DPPC_chol_v28.mat')
    end

    if kk==2
        load('est_retrieval_1.1_v15.mat');
        load('data_for_DPPC_v15.mat');
    end

load('pixOL_com_loc_precision_NFP-700.mat');
sigma_r = sqrt(sigma_x_opt.^2+sigma_y_opt.^2);

phiSlice = linspace(-180,180,19);


count = 0;
for ii=1:1:length(phiSlice)-1
    count = count+1;
    
indx = abs(x_cord)<1300 & abs(y_cord)<1300 & z_cord<1200 & z_cord>0 &SM_est_final(:,5)>1000;
indx2 = indx & phiD_ref<phiSlice(ii+1) & phiD_ref>phiSlice(ii);
r_data = sqrt((x_cord(indx2).^2+y_cord(indx2).^2));

R_est = sqrt(1000^2-(1000-z_cord(indx2)).^2);
r_data = r_data-R_est;

%------------------------------

%subplot(1,35,count); hold on;
%histogram();

N_bins = length(r_data)/20;
edges=linspace((prctile(r_data,2)-50),(prctile(r_data,98)+50),N_bins);
edges_distance = edges(2)-edges(1);
h = histogram(r_data,edges);

h.FaceColor = [128, 255, 0]/255;
h.EdgeColor = 'none';



%----- calculate the full width half maximum ----------------
pdf_data = h.Values; center_data = h.BinEdges;
[max_Data,indx_max] = max(pdf_data);
half_max= max_Data/2;
[~,indx_left] = min(abs(pdf_data(1:indx_max)-half_max));
[~,indx_right] = min(abs(pdf_data(indx_max+1:end)-half_max));
data_FWHM(ii,kk) = center_data(indx_right+indx_max)-center_data(indx_left);

end
end
close all;

Fig1 = figure('Units','inches','InnerPosition',[1,1,3.42,2]);
plot((phiSlice(1:end-1)+phiSlice(2:end))/2,data_FWHM(:,1),'LineWidth',2); hold on;
plot((phiSlice(1:end-1)+phiSlice(2:end))/2,data_FWHM(:,2),'LineWidth',2);
xlabel('\phi (\circ)'); ylabel('FWHM (nm)');
xlim([-180,180]);


set(gcf,'Color','w');
whitebg('w');
legend('DPPC+chol','DPPC');



end