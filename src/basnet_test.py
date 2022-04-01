import os
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms  # , utils
# import torch.optim as optim

import numpy as np
from PIL import Image

from data_loader import *

from src.model import BAS2Net, BASNet


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = os.path.basename(image_name)  # image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def train(model):
    # --------- 1. get image path and name ---------
    # image_dir = './test_data/test_images/'
    # image_dir = './test_data/test_ecssd/'
    # image_dir = './test_data/test_pascal/'
    image_dir = './train_data/DUTS/DUTS-TE/im_aug/'
    if model == "basnet":
        # prediction_dir = './test_data/basnet_50/'
        # prediction_dir = './test_data/basnet_tot/'
        # prediction_dir = './train_data/DUTS/DUTS-TE/rs_basnet_tot/'
        # prediction_dir = './test_data/basnet_50_ecssd/'
        # prediction_dir = './test_data/basnet_50_pascal/'
        # prediction_dir = './test_data/basnet_final_ecssd/'
        # prediction_dir = './test_data/basnet_final_pascal/'
        prediction_dir = './train_data/DUTS/DUTS-TE/rs_basnet/'
        # model_path = './saved_models/basnet_bsi_itr_34500.tar'
        model_path = './saved_models/basnet_bsi_final.tar'
    elif model == "bas2net":
        # prediction_dir = './test_data/bas2net_50/'
        # prediction_dir = './train_data/DUTS/DUTS-TE/rs_bas2net/'
        # prediction_dir = './test_data/bas2net_tot/'
        # prediction_dir = './test_data/bas2net_50_ecssd/'
        # prediction_dir = './test_data/bas2net_50_pascal/'
        # prediction_dir = './test_data/bas2net_final_ecssd/'
        # prediction_dir = './test_data/bas2net_final_pascal/'
        prediction_dir = './train_data/DUTS/DUTS-TE/rs_bas2net_final/'
        # model_path = './saved_models/bas2net_bsi_itr_46500.tar'
        model_path = './saved_models/bas2net_bsi_final.tar'
    else:
        raise ValueError("wrong model")

    img_name_list = glob.glob(image_dir + '*.jpg')
    print("tot images:", len(img_name_list))

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=0)

    # --------- 3. model define ---------
    print("...load " + model + "...")
    if model == "basnet":
        net = BASNet(3, 1)
    elif model == "bas2net":
        net = BAS2Net(3, 1)
    else:
        raise ValueError("wrong model")
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print("epoch:", epoch)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", i_test, img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7, d8


if __name__ == '__main__':
    # train("basnet")
    train("bas2net")
