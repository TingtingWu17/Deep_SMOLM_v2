figure(); 
dens = 0.2;
scatter(bias_con_all(:,1),bias_con_all(:,2),10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
xlim([-50,50]);
ylim([-50,50]);

%%
figure();
histogram(bias_con_all(:,1)); hold on;
histogram(bias_con_all(:,2));
%%
figure();
subplot(1,3,1);
scatter(orien_est_all(:,1),orient_GT_all(:,1),10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
subplot(1,3,2);
scatter(orien_est_all(:,2),orient_GT_all(:,2),10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
subplot(1,3,3);
scatter(orien_est_all(:,3),orient_GT_all(:,3),10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

%%
figure();
scatter(I_GT_all,I_est_all,10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

figure();
histogram(I_est_all-I_GT_all);