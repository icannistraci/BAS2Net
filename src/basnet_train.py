import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import os
import datetime

from data_loader import *
from src.model import BAS2Net
from utils import memory_info, get_last_model, print_model_size

from src import pytorch_ssim, pytorch_iou

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)


def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + ssim_out + iou_out

    return loss


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v):
    loss0 = bce_ssim_loss(d0, labels_v)
    loss1 = bce_ssim_loss(d1, labels_v)
    loss2 = bce_ssim_loss(d2, labels_v)
    loss3 = bce_ssim_loss(d3, labels_v)
    loss4 = bce_ssim_loss(d4, labels_v)
    loss5 = bce_ssim_loss(d5, labels_v)
    loss6 = bce_ssim_loss(d6, labels_v)
    loss7 = bce_ssim_loss(d7, labels_v)
    # ssim0 = 1 - ssim_loss(d0,labels_v)

    # iou0 = iou_loss(d0,labels_v)
    # loss = torch.pow(torch.mean(torch.abs(labels_v-d0)),2)*(5.0*loss0 + loss1 + loss2 + loss3 + loss4 + loss5) #+ 5.0*lossa
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7  # + 5.0*lossa
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (loss0.item(), loss1.item(), loss2.item(),
                                                                               loss3.item(), loss4.item(), loss5.item(),
                                                                               loss6.item()))
    # print("BCE: l1:%3f, l2:%3f, l3:%3f, l4:%3f, l5:%3f, la:%3f, all:%3f\n"%(loss1.data[0],loss2.data[0],loss3.data[0],loss4.data[0],loss5.data[0],lossa.data[0],loss.data[0]))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = './train_data/'
tra_image_dir = 'DUTS/DUTS-TR/im_aug/'
tra_label_dir = 'DUTS/DUTS-TR/gt_aug/'

image_ext = '.jpg'
label_ext = '.png'

in_model_dir = "D:/saved_models/basnet_bsi_1/"
model_dir = "D:/saved_models/basnet_bsi_2/"

log_file = "./logs/" + str(datetime.datetime.now()).replace(" ", "__").replace(".", "_").replace(":", "_").replace("-", "_")\
           + ".log"

epoch_num = 100
curr_epoch = 0
batch_size_train = 8
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    img_name = os.path.basename(img_path)  # img_path.split("/")[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(256),
        RandomCrop(224),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
# define the net
net = BAS2Net(3, 1)
if torch.cuda.is_available():
    net.cuda()
    print("CUDA!")
memory_info("net")
print_model_size(net, False)

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. load model --------
last_model = None
last_epoch = -1
models = glob.glob(in_model_dir + '*.tar')
print(models)
if len(models) != 0:
    last_model = get_last_model(models)
print(last_model)
if last_model is not None:
    checkpoint = torch.load(last_model)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("last epoch:", epoch)
    curr_epoch = epoch - 1

memory_info("net 2")

# ------- 6. training process --------
print("---start training...")

with open(log_file, 'a') as lf:
    lf.write("Start training: " + str(datetime.datetime.now()) + "\n")

ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

for epoch in range(curr_epoch, epoch_num):
    memory_info("epoch" + str(epoch))
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        # memory_info("input size 1, epoch " + str(epoch))

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # memory_info("input size 2, epoch " + str(epoch))
        d0, d1, d2, d3, d4, d5, d6, d7 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, d7, labels_v)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, d7, loss2, loss

        log = "[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val)
        print(log)
        with open(log_file, 'a') as lf:
            lf.write(log + "\n")

        if ite_num % 500 == 0:  # save model every 500 iterations

            # torch.save(net.state_dict(), model_dir + "bas2net_bsi_itr_%d_train_%3f_tar_%3f.pth" % (
            # ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            torch.save({
              'epoch': epoch,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, model_dir + "bas2net_bsi_itr_%d_train_%3f_tar_%3f.tar" % (
            ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

with open(log_file, 'a') as lf:
    lf.write("End training: " + str(datetime.datetime.now()) + "\n")

print('-------------Congratulations! Training Done!!!-------------')

torch.save(net.state_dict(), "bas2net_bsi_final.pth")
torch.save({
              'epoch': epoch_num,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, "bas2net_bsi_final.tar")
