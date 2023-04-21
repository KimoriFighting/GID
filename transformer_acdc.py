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
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    from datasets.dataset_acdc import Acdc_dataset
    db_train = Acdc_dataset(split="train",
                           transform=transforms.Compose(
                               [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    # db_train = Synapse_dataset(base_dir=args.train_path, list_dir=args.list_dir, split="train",
    #                            transform=transforms.Compose(
    #                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    db_test = Acdc_dataset(split="test",
                            transform=transforms.Compose(
                                [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    # for i_batch, sampled_batch in enumerate(trainloader):
    #     image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
    model = TUNet(num_classes=args.num_classes, vt_type=args.vit_name, img_size=args.img_size, middle=[9, 14, 19]).to(device)
    model.load_from(args.pre_path)
    # state = torch.load('/search/hadoop04/jiangxinfa/yongsen/GID/save/transformerL_UDF_199.pth',
    #                    map_location=device)
    # trans = state['model']
    # model.load_state_dict(trans)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = LR_Scheduler_Head(args.lr_scheduler, args.base_lr,
                                  args.epochs, len(trainloader), warmup_epochs=5)

    first = time.time()
    print(first)
    for epoch in range(args.epochs):
        train(trainloader, model, ce_loss, dice_loss, optimizer, device, scheduler, epoch)
    end = time.time()
    print(end)
    #
    # # save weights
    # save_files = {
    #     'backbone': model.backbone.state_dict(),
    #     'model': model.state_dict(),
    # }
    # torch.save(save_files,
    #            "./save/transformerL_UDF_{}.pth".format(
    #                epoch))

    # val(args, testloader, model, len(db_test))
    val_acdc(model, testloader, device)
    # print(first-end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str,
                        default='/search/hadoop04/jiangxinfa/yongsen/TransUnet/project_TransUNet/data/Synapse/train_npz',
                        help='root dir for data')
    parser.add_argument('--test_path', type=str,
                        default='/search/hadoop04/jiangxinfa/yongsen/TransUnet/project_TransUNet/data/Synapse/test_vol_h5',
                        help='root dir for data')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    # parser.add_argument('--pre_path', type=str,
    #                     default='/search/hadoop04/jiangxinfa/yongsen/GID/save/imagenet21k_ViT-L_16.npz')
    parser.add_argument('--pre_path', type=str,
                        default='/mnt/disk/yongsen/model/vit/imageNet21k/imagenet21k_ViT-L_16.npz')
    parser.add_argument('--cuda', default='cuda:0', help='disables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='learning rate scheduler (default: poly)')
    parser.add_argument('--epochs', type=int,
                        default=10, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='ViT-L_16', help='select one vit model', choices=['ViT-B_16, ViT-L_16'])
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')

    args = parser.parse_args()
    set_seed(args)
    main(args)