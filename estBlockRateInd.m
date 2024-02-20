% function out = estBlockRateInd(hist_out, y_sub_image, repeat_height, repeat_width)
% %     y_sub_dct = reshape(y_sub_image, 8, 8)';
%     y_sub_dct = y_sub_image;
%     nbits = 0;
%     temp = log2(repeat_height*repeat_width);
%     for i = 1:1:8
%         for j = 1:1:8
%             value = y_sub_dct(i,j);
%             nelements = hist_out{i,j}{1,1};
%             ncenters = hist_out{i,j}{1,2};
%             index = value - ncenters(1) + 1;
%             nbits = nbits + (temp - log2(nelements(index)));
%         end
%     end
%     out = nbits;
% end

function out = estBlockRateInd(y_hist_out, cb_hist_out,cr_hist_out,repeat_height, repeat_width)
%     y_sub_dct = reshape(y_sub_image, 8, 8)';
    rate_out = zeros(8,8,255,2);
    temp = log2(repeat_height*repeat_width);
    for i = 1:1:8
        for j = 1:1:8
            y_nelements = y_hist_out{i,j}{1,1};
            y_ncenters = y_hist_out{i,j}{1,2};
            cb_nelements = cb_hist_out{i,j}{1,1};
            cb_ncenters = cb_hist_out{i,j}{1,2};
            cr_nelements = cr_hist_out{i,j}{1,1};
            cr_ncenters = cr_hist_out{i,j}{1,2};
            y_len = size(y_ncenters,2);
            cb_len = size(cb_ncenters,2);
            cr_len = size(cr_ncenters,2);
            for qpstep = 1:255
                y_rate = 0;
                cb_rate = 0;
                cr_rate = 0;

                y_nquant = round(y_ncenters/qpstep);
                cb_nquant = round(cb_ncenters/qpstep);
                cr_nquant = round(cr_ncenters/qpstep);
                for m = y_nquant(1):y_nquant(end)

                    count = sum(y_nelements(y_nquant == m));
                    
                    if count ~= 0
                        bpp = temp - log2(count);
                        if bpp == 0
                            bpp = 3.0/64;
                        end
                        y_rate = y_rate + count*bpp;
                    end
                end
                for m =cb_nquant(1):cb_nquant(end)
                     count = sum(cb_nelements(cb_nquant == m));
                    
                    if count ~= 0
                        bpp = temp - log2(count);
                        if bpp == 0
                            bpp = 3.0/64;
                        end
                        cb_rate = cb_rate + count*bpp;
                    end


                end
                for m =cr_nquant(1):cr_nquant(end)

                  count = sum(cr_nelements(cr_nquant == m));
                    
                    if count ~= 0
                        bpp = temp - log2(count);
                        if bpp == 0
                            bpp = 3.0/64;
                        end
                        cr_rate = cr_rate + count*bpp;
                    end

                end
                rate_out(i,j,qpstep,1) = y_rate;
                rate_out(i,j,qpstep,2) = cb_rate + cr_rate;
            end
        end
    end
    out = rate_out;
end