function [rotation, translation] = transform(ref_name, in_name)

    ref = load_untouch_nii(ref_name);
    in = load_untouch_nii(in_name);    
    ref = squeeze(ref.img);   
    in = squeeze(in.img);    

    optimizer = registration.optimizer.RegularStepGradientDescent;    
    metric = registration.metric.MattesMutualInformation;
        
    
    mat1 = imregtform(in, ref, 'rigid', optimizer, metric);    
    mat = mat1.T;    
    mat_t = mat';
        
    res = [mat_t(1) mat_t(2) mat_t(3);
        mat_t(5) mat_t(6) mat_t(7);
        mat_t(9) mat_t(10) mat_t(11);
        ];
    
    rotation = get_rot_axis(res);
    translation = [mat_t(13) mat_t(14) mat_t(15)];


    % disp(angle);
end    
    
    
    
    
    
    
    
    
    
    
    
    
    
    