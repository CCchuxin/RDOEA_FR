function out = GetWholeCoffHist(in, repeat_height, repeat_width)
    y_sub_dct = in;
    y_compressed_img = zeros(8*repeat_height, 8*repeat_width);
    for i = 1:repeat_height
        for j = 1:repeat_width
            y_compressed_img(((i-1)*8+1):(i*8), ((j-1)*8+1):(j*8)) = (reshape(y_sub_dct{i,j},8,8))';
        end
    end
    
    y_elem = reshape(y_compressed_img, repeat_height*repeat_width*8*8, 1);
    max_value = max(y_elem);
    min_value = min(y_elem);
   [nelements, ncenters] = hist(y_elem, [min_value:max_value]); 
   hist_out = {nelements,ncenters};
     
    out = hist_out;
   
end