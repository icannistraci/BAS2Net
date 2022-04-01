import torch
import gc
from functools import reduce
import glob
import shutil
import os

def memory_usage(name):
    num = 0
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if len(obj.size()) > 0:
                    num += 1
                    if obj.type() == 'torch.cuda.FloatTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
                    elif obj.type() == 'torch.cuda.LongTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 64
                    elif obj.type() == 'torch.cuda.IntTensor':
                        total += reduce(lambda x, y: x * y, obj.size()) * 32
        except:
            pass
    tot = "{} GB".format(total / ((1024 ** 3) * 8))
    print("MEMORY USAGE", name, "-", tot, "-", num, "\n")


def memory_info(name):
    alloc = torch.cuda.memory_allocated()
    cache = torch.cuda.memory_cached()
    res = "{} GB allocated, {} GB cache".format(alloc / (1024 ** 3), cache / (1024 ** 3))
    print("MEMORY INFO", name, "-", res, "\n")


def print_model_size(model, disp=True):
    model_sz = 0
    for name, param in model.named_parameters():
        if disp:
            print(name, param.size())
        if param.requires_grad:
            model_sz += param.numel()
    print("Model size:", model_sz)
    return model_sz


def image_subset(dir):
    img_src_dir = os.path.join(dir, "im_aug")
    img_dst_dir = os.path.join(dir, "im_aug_new")
    mask_src_dir = os.path.join(dir, "gt_aug")
    mask_dst_dir = os.path.join(dir, "gt_aug_new")

    print(os.listdir(img_src_dir))

    count = 0
    for jpgfile in glob.iglob(os.path.join(img_src_dir, "*.jpg")):

        if count < 1000:
            name, ext = os.path.splitext(os.path.basename(jpgfile))
            shutil.copy(jpgfile, img_dst_dir)
            shutil.copy(os.path.join(mask_src_dir, name + ".png"), mask_dst_dir)
            count = count + 1
        else:
            break


def get_last_model(models):
    model = None
    last_it = -1
    for m in models:
        it = int(os.path.basename(m).split("_")[3])
        if it > last_it:
            model = m
            last_it = it
    return model


def move_pascal(src, dst, ext):
    imgs = glob.glob(src + "/*/*."+ext)
    print(len(imgs))
    for i, img in enumerate(imgs):
        name = str(i) + "." + ext
        shutil.copyfile(img, os.path.join(dst, name))
        print("copied", img, os.path.join(dst, name))


if __name__ == '__main__':
    # import torchvision
    # net = torchvision.models.resnet34()
    # print_model_size(net, False)
    #
    # net = torchvision.models.resnet50()
    # print_model_size(net, False)
    #
    # mydir = "./train_data/DUTS/DUTS-TE"
    # image_subset(mydir)

    # model_dir = "./saved_models/basnet_bsi/"
    # models = os.listdir(model_dir)
    # print(get_last_model(models))

    # move_pascal("C:\\Users\super\OneDrive\\universita\\aml - Galasso\AML-19-20\Project\Pascal\datasets\imgs",
    #             "./test_data/test_pascal", "jpg")
    move_pascal("C:\\Users\super\OneDrive\\universita\\aml - Galasso\AML-19-20\Project\Pascal\datasets\masks",
                "./test_data/gt_pascal", "png")

