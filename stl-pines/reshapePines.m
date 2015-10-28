function [resultSet] = reshapePines(pines, bands, gt)
if(gt==true)
    [x, y ] = size(pines);
    resultSet  = zeros(x*y,1);
    index = 1;
   for j = 1:y
    for i = 1 : x
        resultSet(index) = pines(i,j);
        index = index +1;
    end
    end
else
    [x, y, ~ ] = size(pines);
    resultSet  = zeros(x*y, bands);
    index = 1;
    for j = 1:y
        for i = 1 : x
            yy = pines(i,j,1:bands);
        resultSet(index,:) = yy(:);
        index = index +1;
        end
    end
end


end