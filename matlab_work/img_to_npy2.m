path = ['D:\Xuanhang_file\nii\'];

save_data_path =  ['D:\Xuanhang_file\train_data\'];
save_label_path = ['D:\Xuanhang_file\train_label\'];

img_save = [];

index = 1;

% 注意check范围
for i=19:19
    if i < 10
        file_path = [path 'DB0' num2str(i)];
    else
        file_path = [path 'DB' num2str(i)];
    end

    ref_path = [file_path '\0.nii'];
    ref = load_untouch_nii(ref_path).img;
    for j=1:159
        mov_path = [file_path '\' num2str(j) '.nii'];
        mov = load_untouch_nii(mov_path).img;

        mov = imhistmatchn(mov, ref);

        img_stack = [];

        img_stack(:,:,:,1) = ref;
        img_stack(:,:,:,2) = mov;

        img_save(index, :,:,:,:) = img_stack;
        index = index + 1;
    end
end
writeNPY(img_save, 'test_data_align.npy')
