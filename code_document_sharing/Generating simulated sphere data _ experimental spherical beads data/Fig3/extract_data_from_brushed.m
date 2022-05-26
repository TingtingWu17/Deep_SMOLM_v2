function data = extract_data_from_brushed(brushedData, rawData, infoIndx)

data = [];
for ii = 1:size(brushedData,1)
    indx_cur = find(brushedData(ii,1)==rawData(:,infoIndx(1)) & ...
                    brushedData(ii,2)==rawData(:,infoIndx(2)) & ...
                    brushedData(ii,3)==rawData(:,infoIndx(3)));
    data = [data; rawData(indx_cur,:)];       
end
      
end