path = 'E:\read_nii\nii\';
save_data_path = 'E:\motion_sim_data\nii\';
save_param_path = 'E:\motion_sim_data\param\';
for i=3:19
    if i < 10
        dir_path = [path 'DB0' num2str(i)];
        save_data_name = [save_data_path '\DB0' num2str(i)];
        save_param_name = [save_param_path '\DB0' num2str(i)];

    else
        dir_path = [path 'DB' num2str(i)];
        save_data_name = [save_data_path '\DB' num2str(i)];
        save_param_name = [save_param_path '\DB' num2str(i)];
    end
    for j=0:159
        save_data_dir = [save_data_name '_' num2str(j)];
        save_param_dir = [save_param_name  '_' num2str(j)];
        if ~exist(save_data_dir)
            mkdir(save_data_dir)
        end
        if ~exist(save_param_dir)
            mkdir(save_param_dir)
        end
        mov_name = [dir_path '\' num2str(j) '.nii'];
        generate_motion_sim(mov_name, save_data_dir, save_param_dir, 32, 2,5);
    end
end
