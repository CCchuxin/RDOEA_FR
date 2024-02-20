function out = GetInvCoffHist(in, repeat_height, repeat_width)
%   lumMat1 = [
%                     16 11 10 16 24 40 51 61 ...
%                     12 12 14 19 26 58 60 55 ...
%                     14 13 16 24 40 57 69 56 ...
%                     14 17 22 29 51 87 80 62 ...
%                     18 22 37 56 68 109 103 77 ...
%                     24 35 55 64 81 104 113 92 ...
%                     49 64 78 87 103 121 120 101 ...
%                     72 92 95 98 112 100 103 99];
%                 lumMat1=reshape(lumMat1,8,8);%lp
    y_sub_dct = in;
    y_compressed_img = zeros(8*repeat_height, 8*repeat_width);
    for i = 1:repeat_height
        for j = 1:repeat_width
%             y_compressed_img(((i-1)*8+1):(i*8), ((j-1)*8+1):(j*8)) = (reshape(y_sub_dct{i,j},8,8))';
                y_compressed_img(((i-1)*8+1):(i*8), ((j-1)*8+1):(j*8)) = y_sub_dct{i,j};
        end
    end
    y_compressed_img = round(y_compressed_img);
    
    hist_out = cell(8,8);
    
    for i = 1:1:8
        for j = 1:1:8
            elem=y_compressed_img(i:8:(8*repeat_height),j:8:(8*repeat_width));
            y_elem = reshape(elem, repeat_height*repeat_width, 1); 
%                y_elem=round(y_elem/lumMat1(i,j));%lp
            max_value = max(y_elem);
            min_value = min(y_elem);
            
            if max_value ~= min_value 
                [nelements, ncenters] = hist(y_elem, [min_value:max_value]);
            else
                ncenters = max_value;
                nelements = double(repeat_height*repeat_width/(2^(3.0/64)));
            end
            hist_out{i,j} = {nelements,ncenters};
%             histout11=[ncenters;nelements];
% %             dlmwrite('histout11.txt',histout11);
%             figure;
%             plot(ncenters,nelements);
%             bar(ncenters,nelements);
%              xlabel('transform  coefficients');ylabel('counts');
%             [yi,xi]=ksdensity(y_elem);
%             laplace=fittype(@(a,b,x) 1./(2*b)*exp(-abs(x-a)./b));
%             laplacefit=fit(xi',yi',laplace);
%             yy=laplacefit(xi);
%             a=laplacefit.a;
%             b=laplacefit.b;  
        end
    end
     
    out = hist_out;   %lp

end