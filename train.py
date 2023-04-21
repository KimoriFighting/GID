import argparse
from torchvision import transforms
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim

from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from models import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    db_train = Synapse_dataset(base_dir=args.train_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    # testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    model = LCLT_Net(vt_type=args.vit_name).to(device)
    teacher = Teacher(vt_type=args.vit_name).to(device)
    state = torch.load(args.pre_path,
               map_location=device)
    trans = state['backbone']
    teacher.backbone.load_state_dict(trans)


    ce_loss = CrossEntropyLoss().to(device)
    dice_loss = DiceLoss(args.num_classes).to(device)
    kl_loss = KLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = LR_Scheduler_Head(args.lr_scheduler, args.base_lr,
                                  args.epochs, len(trainloader), warmup_epochs=5)
    # model.load_state_dict(
    #     torch.load('/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/Student/transformer0_unet_199.pth')[
    #         'model'])
    #
    # val(args, testloader, model, len(db_test))
    loss_save = []
    kl_save = []
    for epoch in range(args.epochs):
        loss, kl = train_stu(trainloader, model, teacher, ce_loss, dice_loss, kl_loss, optimizer, device, scheduler, epoch, args)
        loss_save.append(loss)
        kl_save.append(kl)

    # save weights
    save_files = {
        'backbone': model.backbone.state_dict(),
        'decoder': model.decoder.state_dict(),
        'model': model.state_dict(),
    }
    torch.save(save_files,
               "/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/Student/{}".format(
                   args.save_name))

    with open("L.txt", "w") as output:
        for i in loss_save:
            output.write(str(i) + '\n')

    with open("KL.txt", "w") as output:
        for i in kl_save:
            output.write(str(i) + '\n')

    # val(args, testloader, model, len(db_test))

def test(args):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = Student().to(device)

    state = torch.load('/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/Student/{}'.format(
                   args.save_name), map_location=device)
    backbone = state['backbone']
    decoder = state['decoder']
    model.backbone.load_state_dict(backbone)
    model.decoder.load_state_dict(decoder)

    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    val(args, testloader, model, len(db_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str,
                        default='/home/yongsen/Segments/TransUNet/project_TransUNet/data/Synapse/train_npz',
                        help='root dir for data')
    parser.add_argument('--test_path', type=str,
                        default='/home/yongsen/Segments/TransUNet/project_TransUNet/data/Synapse/test_vol_h5',
                        help='root dir for data')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')

    # 经常该
    parser.add_argument('--pre_path', type=str,
                        default='/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/Teacher/transformer_unet_149.pth')
    # parser.add_argument('--pre_path', type=str,
    #                     default='/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/Teacher/transformer_s_unet_149.pth')
    parser.add_argument('--vit_name', type=str,
                        default='ViT-L_16', help='select one vit model', choices=['ViT-B_16, ViT-L_16'])
    parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
    parser.add_argument('--n_skip', type=int,
                        default=1, help='using number of skip-connect, default is num')
    parser.add_argument('--save_name', type=str,
                        default='lclt1_transformerL_unet10z3002')


    parser.add_argument('--cuda', default='cuda:0', help='disables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='learning rate scheduler (default: poly)')
    parser.add_argument('--epochs', type=int,
                        default=300, help='maximum epoch number to train')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--base_lr', type=float, default=0.02,
                        help='segmentation network learning rate')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')

    args = parser.parse_args()
    set_seed(args)
    for i in range(10):
        main(args)
        test(args)