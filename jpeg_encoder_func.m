function outData = jpeg_encoder_func (y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, ~, lum_quant,chrom_quant, fast_mode, mse_table, rate_table)
global zigZagOrder;    
fid = 0;
    rate_block = zeros(repeat_height, repeat_width);
     mse_block = zeros(repeat_height, repeat_width);
    y_bits = 0;
    rate_block_est = zeros(repeat_height, repeat_width);
    y_sub_image_iq =  y_sub_image;
    mse = 0.0;
    mse_est = 0.0;
    rate_est = 0;
    if fast_mode ~= 1
        lastDC(1) = 0;
        lastDC(2) = 0;
        lastDC(3) = 0;
        for i=1:repeat_height
            for j=1:repeat_width
                y_sub_image{i, j} = quantize(y_sub_image{i, j}, 'lum');
%                 y_sub_image_block = y_sub_image{i, j}; 
                y_sub_image_iq{i, j} = iquantize(y_sub_image{i, j},'lum');
                [lastDC(1), y_bits] = huffman(fid, y_sub_image{i, j}, lastDC(1), 1, 1);
                 
                cb_sub_image{i, j} = quantize(cb_sub_image{i, j}, 'chrom');
                cb_sub_image_iq{i,j} = iquantize(cb_sub_image{i,j},'chrom');
                [lastDC(2), cb_bits]= huffman(fid, cb_sub_image{i, j}, lastDC(2), 2, 2);

                cr_sub_image{i, j} = quantize(cr_sub_image{i, j}, 'chrom');
                cr_sub_image_iq{i,j} = iquantize(cr_sub_image{i,j},'chrom');
                [lastDC(3), cr_bits] = huffman(fid, cr_sub_image{i, j}, lastDC(3), 2, 2);
                rate_block(i,j) = rate_block(i,j) + y_bits+cb_bits+cr_bits;                
                dct = y_sub_dct{i, j}+cb_sub_dct{i,j}+cr_sub_dct{i,j};
                idct =  y_sub_image_iq{i, j}+cb_sub_image_iq{i,j}+cr_sub_image_iq{i,j};
                mse_block(i,j) = sum(sum(((dct - idct ).^2)));
                mse = mse + sum(sum(((dct - idct ).^2)));
            end
        end
         real=reshape(rate_block,repeat_height*repeat_width,1);
%         dlmwrite('real_rate.txt',real);
    else
        lum_mse_est = 0;
        lum_rate_est = 0;
        chrom_mse_est = 0;
        chrom_rate_est = 0;
        mse_est = 0;
        rate_est = 0;
        lumMat = reshape(lum_quant, 8, 8)';
        chromMat = reshape(chrom_quant, 8, 8)';
        
        for i = 1:8
            for j = 1:8
                lum_mse_est = lum_mse_est + mse_table(i,j,lumMat(i,j),1);
                chrom_mse_est = chrom_mse_est + mse_table(i,j,chromMat(i,j),2);             
                mse_est = lum_mse_est + chrom_mse_est;
                
                lum_rate_est = lum_rate_est + rate_table(i,j,lumMat(i,j),1);
                chrom_rate_est = chrom_rate_est + rate_table(i,j,chromMat(i,j),2);
                rate_est = lum_rate_est + chrom_rate_est;
            end
        end  
    end
    sum(sum(mse_block));
    outData = [mse/(repeat_width*8*repeat_height*8),sum(sum(rate_block))/(repeat_width*8*repeat_height*8), mse_est/(repeat_width*8*repeat_height*8) ,rate_est/(repeat_width*8*repeat_height*8)];
end
  