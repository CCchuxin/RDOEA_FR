function out = GetInvLapCoffHist(in, repeat_height, repeat_width)
    y_sub_dct = in;
    y_compressed_img = zeros(8*repeat_height, 8*repeat_width);
    
    global quantLumMatrix;
   
    
    for i = 1:repeat_height
        for j = 1:repeat_width
            y_compressed_img(((i-1)*8+1):(i*8), ((j-1)*8+1):(j*8)) = y_sub_dct{i,j} ;%(reshape(y_sub_dct{i,j},8,8))'
        end
    end
    
    hist_out = cell(8,8);
    
    for i = 1:1:8
        for j = 1:1:8
            
            if i == 1 && j == 1
                last_dc = 0;
                for m = 1:repeat_height
                    for n = 1:repeat_width
                        
                        y_compressed_img(8*(m - 1) + 1, 8*(n - 1) + 1) =  y_compressed_img(8*(m - 1) + 1, 8*(n - 1) + 1) - last_dc;
                        last_dc = y_compressed_img(8*(m - 1) + 1, 8*(n - 1) + 1) + last_dc;
                    end
                end     
            end
            
            qpstep = quantLumMatrix((i - 1)*8+j);
            elem=y_compressed_img(i:8:(8*repeat_height),j:8:(8*repeat_width));
            y_elem = (reshape(elem, repeat_height*repeat_width, 1)); 
            max_value = ceil(max(y_elem));
            min_value = floor(min(y_elem));
            max_index = round((max_value)/qpstep);
            min_index = round((min_value)/qpstep);
            
            [nelements, ncenters] = hist(y_elem,min_value:max_value);
            lamda = repeat_height * repeat_width/(2*nelements(1-ncenters(1)));
            miu = 0;
%             [yi,xi]=ksdensity(y_elem);
            laplace=fittype(@(a,b,x) 1./(2*b)*exp(-abs(x-a)./b));
%             laplacefit=fit(xi',yi',laplace, 'StartPoint', [miu,lamda]);
            yi = nelements/(repeat_width*repeat_height);
            xi = ncenters;
            laplacefit=fit(xi',yi',laplace, 'StartPoint', [miu,lamda]);
            yy=laplacefit(xi);
            a=laplacefit.a;
            b=laplacefit.b; 


            
            nelements = zeros(max_value - min_value + 1, 1);
            ncenters = zeros(max_value - min_value + 1, 1);
            
            if max_index ~= min_index 
                for m = 1:(max_value - min_value + 1)
                    index = round((m + (min_value - 1))/qpstep);
                    if(index == 0)
%                         nelements(m) = laplacefit(double(qpstep/4))*qpstep;
%                         nelements(m) = 1./(2*b)*exp(-abs(qpstep/4-a)./b) *qpstep;
                        nelements(m) = 1.0 - exp((-qpstep/2 - a)/b)/2 - exp((a - qpstep/2)/b)/2;
                        ncenters(m) = m + (min_value - 1);
                    else
%                         nelements(m) = laplacefit(double(index*qpstep))*qpstep;
%                         nelements(m) = 1./(2*b)*exp(-abs(double(index*qpstep)-a)./b) *qpstep;
                        if(index < 0)
                            nelements(m) = exp((index*qpstep+qpstep/2 - a)/b)/2 - exp((index*qpstep - qpstep/2 - a)/b)/2;
                        else
                            nelements(m) = exp((a - index*qpstep + qpstep/2)/b)/2 - exp((a - index*qpstep - qpstep/2 )/b)/2;
                        end
                        ncenters(m) = m + (min_value - 1);
                    end
                end
            else
                ncenters(:) = min_value:max_value;
                nelements(:) = 1.0 - exp((-qpstep/2 - a)/b)/2 - exp((a - qpstep/2)/b)/2;;
            end
            
            hist_out{i,j} = {nelements,ncenters};
             
        end
    end
     
    out = hist_out;
   
end