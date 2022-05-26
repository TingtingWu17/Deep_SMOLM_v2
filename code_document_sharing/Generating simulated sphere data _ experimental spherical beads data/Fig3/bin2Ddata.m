function hotN = bin2Ddata(data_para1,data_para2,edges_para1,edges_para2)

Y1 = discretize(data_para1,edges_para1);
Y2 = discretize(data_para2,edges_para2);

hotN = zeros(length(edges_para1),length(edges_para2));
for ii = 1:length(edges_para1)
    for jj = 1:length(edges_para2)
    edge_cur = edges_para1(ii);
    hotN(ii,jj) = sum(Y1==ii&Y2==jj);
    end
    
end

end