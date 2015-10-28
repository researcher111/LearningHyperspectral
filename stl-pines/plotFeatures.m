function  plotFeatures(matrix_input, num)
%this functions generates collection of subplots representing the features
%in 

if(num)
    numsqrt = round(sqrt(num)); 
    for i = 1:num
            subplot(numsqrt,numsqrt+1,i);
            plot(matrix_input(i,:));
    end
else
    [x, ~ ] = size(matrix_input);

    xsqrt = round(sqrt(x))+1;
    for i = 1:x
            subplot(xsqrt,xsqrt+1,i);
            plot(matrix_input(i,:));
    end
end
end

