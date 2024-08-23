file_path = ['test\nii'];

reg_res_save = [];

index = 1;
data = [];


% 注意check 范围

ref_path = [file_path '\0.nii'];
for j=1:32
    mov_path = [file_path '\' num2str(j) '.nii'];
    [rot, tran] = transform(ref_path, mov_path);

    data = [tran, rot];

    reg_res_save(:, index) = data;
    index = index + 1;


end

writeNPY(reg_res_save, 'reg_res.npy')
