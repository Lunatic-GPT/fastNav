import torch
from loupe_torch import model, dataloader, train_utils
from torch.utils.data import DataLoader
import numpy as np
import os
import time

device = 'cuda'
learning_rate = 1e-5
total_epochs = 500
bs = 17

pmask_slope = 200
sample_slope = 5
sparsity = 0.03125
eps = 0.01

train_path = '/public/home/diaoxh2022/fast_FatNav/data/train_set/train_data.npy'
label_path = '/public/home/diaoxh2022/fast_FatNav/data/train_set/train_label.npy'

#train_path = 'D:/Xuanhang_file/dual_loss_lazy_new/train_set/train_data.npy'
#label_path = 'D:/Xuanhang_file/dual_loss_lazy_new/train_set/train_label.npy'
test_data_path = '/public/home/diaoxh2022/fast_FatNav/data/test_set/test_data.npy'
test_label_path = '/public/home/diaoxh2022/fast_FatNav/data/test_set/test_label.npy'
# test_path = ''

#test_data_path = 'D:/Xuanhang_file/dual_loss_lazy_new/test_data/test_data.npy'
#test_label_path = 'D:/Xuanhang_file/dual_loss_lazy_new/test_data/test_label.npy'
# test_path = ''
save_path = 'saved_model_large'
mask_path = 'saved_mask_large'

train_set = dataloader.Dataset(train_path, label_path)
train_dataloader = DataLoader(train_set, batch_size=bs)

test_set = dataloader.Dataset(test_data_path, test_label_path)
test_dataloader = DataLoader(test_set, batch_size=bs)

if not os.path.exists(save_path):
    os.mkdir(save_path)

if not os.path.exists(mask_path):
    os.mkdir(mask_path)

dim = torch.Size([80, 80, 80, 1])

my_model = model.Loupe(image_dims=dim, pmask_slope=pmask_slope, sample_slope=sample_slope, sparsity=sparsity,
                       device=device,
                       eps=eps).to(device)

layer_name = []
for name, param in my_model.named_parameters():
    layer_name.append(name)

criterion = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(my_model.parameters(), lr=learning_rate, weight_decay=0.95)

filename = os.path.join(save_path, 'model{epoch:02d}.pth')
maskname = os.path.join(mask_path, 'mask{epoch:02d}.npy')
train_loss = []
valid_loss = []
train_epochs_loss = []

for epoch in range(0, total_epochs):
    s_time = time.time()
    my_model.train()
    train_epoch_loss = []
    for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
        data_x = data_x.to(torch.float32).to(device)
        data_y = data_y.view(data_y.shape[0], -1)
        data_y = data_y.to(torch.float32).to(device)

        image_in, image_out, param_rot, param_trans = my_model(data_x)

        if epoch < 300:
            for name, param in my_model.named_parameters():
                for item in layer_name[59:]:
                    if name == item:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            loss = criterion(image_in, image_out)
        else:
            for name, param in my_model.named_parameters():
                for item in layer_name[0:59]:
                    if name == item:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            loss1 = criterion(data_y[:, 3:], param_trans)
            loss2 = criterion(data_y[:, 3:], param_rot)
            loss = loss1 + loss2

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, my_model.parameters()), lr=learning_rate)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
    e_time = time.time()
    print("training time=" + str(e_time - s_time))
    if epoch % 10 == 0:
        avg = np.average(train_epoch_loss)
        train_epochs_loss.append(avg)

        my_model.eval()
        test_epochs_loss = []
        with torch.no_grad():
            for idx, (data_x, data_y) in enumerate(test_dataloader, 0):
                data_x = data_x.to(torch.float32).to(device)
                data_y = data_y.view(data_y.shape[0], -1)
                data_y = data_y.to(torch.float32).to(device)

                image_in, image_out, param_rot, param_trans = my_model(data_x)
                if epoch < 300:
                    loss_test = criterion(image_in, image_out).cpu()
                else:
                    loss_test1 = criterion(data_y[:, 3:], param_trans).cpu()
                    loss_test2 = criterion(data_y[:, 3:], param_rot).cpu()
                    loss_test = loss_test1 + loss_test2
                test_epochs_loss.append(loss_test.item())

        loss_test_avg = np.average(test_epochs_loss)
        print("average train loss in epoch " + str(epoch) + "=" + str(avg))
        print("average eval loss in epoch " + str(epoch) + "=" + str(loss_test_avg))
        if epoch >= 300:
            print("eval loss in epoch " + str(epoch) + ", trans_loss=" + str(loss_test1.item()) + ", rot_loss=" + str(loss_test2.item()))
        torch.save(my_model, filename.format(epoch=epoch))
        torch.cuda.empty_cache()
        mask = my_model.state_dict()['pmask']
        np.save(maskname.format(epoch=epoch), mask.detach().cpu().numpy())
