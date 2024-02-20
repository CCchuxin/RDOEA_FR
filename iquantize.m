function out = iquantize(in, code)

    global quantLumMatrix;
    global quantChromMatrix;
    if strcmp(code, 'lum')
        outputData = in .* quantLumMatrix;
    elseif strcmp(code, 'chrom')
        outputData = in .* quantChromMatrix;
    end     

    out = outputData;
end