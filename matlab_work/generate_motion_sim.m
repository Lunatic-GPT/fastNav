function generate_motion_sim(image_name, save_nii_path, save_param_path, num_of_volume, trans_bias, rot_bias)
img = load_untouch_nii(image_name).img;


nii_ref = make_nii(img);
save_nii(nii_ref,[save_nii_path '\0.nii']);

% img_pad = padarray(img, [23, 27, 25]);
% img_pad = img;
% first implied translation, then applied rotation
img_pad = padarray(img, [11,15,13]); % padding to 96x96x96


nii_ref = make_nii(img_pad);
save_nii(nii_ref,[save_nii_path '\0.nii']);



% rot = randn(num_of_volume*3, 1);
rot = dual_peak_gaussian(rot_bias, num_of_volume*3);
randIndex_rot = randperm(size(rot,1));
rot = rot(randIndex_rot, :);


% trans = rand(num_of_volume*3, 1);
trans = dual_peak_gaussian(trans_bias, num_of_volume*3);
randIndex_trans = randperm(size(trans, 1));
trans = trans(randIndex_trans, :);

% save_nii_path = ['test\nii'];
% save_param_path = ['test\param'];

if ~exist(save_param_path)
    mkdir(save_param_path);
end
if ~exist(save_nii_path)
    mkdir(save_nii_path);
end
for i=1:num_of_volume
    rot_x = rot((i-1)*3 + 1);
    rot_y = rot((i-1)*3 + 2);
    rot_z = rot(i*3);

    trans_x = trans((i-1)*3 + 1);
    trans_y = trans((i-1)*3 + 2);
    trans_z = trans(i*3);

    params = [trans_x trans_y trans_z; rot_x rot_y rot_z];
    translation = [trans_x trans_y trans_z];
    rotation = [rot_x rot_y rot_z];

    tform = rigidtform3d(rotation,translation);
    sameAsInput = affineOutputView(size(img_pad),tform,"BoundsStyle","SameAsInput");


    save_path_curr_nii = [save_nii_path '\' num2str(i) '.nii'];
    save_path_curr_param = [save_param_path '\' num2str(i) '.mat'];

    res_volume = imwarp(img_pad,tform,"OutputView",sameAsInput);


    nii = make_nii(res_volume);
    save(save_path_curr_param,'params')
    save_nii(nii,save_path_curr_nii);
    %view_nii(make_nii(mriVolumeTranslated));
end
end