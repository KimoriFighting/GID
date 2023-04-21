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
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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

    db_test = Synapse_dataset(base_dir=args.test_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
        int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    # state = torch.load('/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/TransUNet/transunet1_149.pth',
    #                    map_location=device)
    # trans = state['model']
    # model.load_state_dict(trans)
    model.load_from(weights=np.load(config_vit.pretrained_path))
    model.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    scheduler = LR_Scheduler_Head(args.lr_scheduler, args.base_lr,
                                  args.epochs, len(trainloader), warmup_epochs=5)


    for epoch in range(args.epochs):
        train(trainloader, model, ce_loss, dice_loss, optimizer, device, scheduler, epoch)

    val(args, testloader, model, len(db_test))

    # save weights
    save_files = {
        'model': model.state_dict(),
    }
    torch.save(save_files,
               "/mnt/disk/yongsen/code/Semantic_Segmentation/LCLT/TransUNet/transunet1_{}.pth".format(
                   epoch))

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
    parser.add_argument('--cuda', default='cuda:0', help='disables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='learning rate scheduler (default: poly)')
    parser.add_argument('--epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
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
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')

    args = parser.parse_args()
    set_seed(args)
    main(args)