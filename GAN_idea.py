import argparse
from torchvision import transforms
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
import torch.nn.functional as F
from datasets.dataset_game import *
from models import *
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    db_train = HemangiomaDataset(r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/img',
                                 r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/mask',
                                 opt=transforms.Compose([
                                     resize(size=(args.img_size, args.img_size)),
                                     random_scale_crop(range=(0.75, 1.25)),
                                     random_flip(lr=True, ud=True),
                                     random_rotate(range=(0, 359)),
                                     random_image_enhance(),
                                     random_gaussian_blur(),
                                     random_dilation_erosion(kernel_range=(2, 5)),
                                     tonumpy(),
                                     normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                     totensor(),
                                 ])
                                 )
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                             worker_init_fn=worker_init_fn)

    for i, sample in enumerate(trainloader):
        train_image, train_label = sample['image'], sample['label']

    model = GAN_Uet(num_classes=args.num_classes, vt_type=args.vit_name, img_size=args.img_size, middle=[9, 14, 19]).to(device)

    D_A = Discriminator((3, args.img_size, args.img_size)).to(device)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = LR_Scheduler_Head(args.lr_scheduler, args.base_lr,
                                  args.epochs, len(trainloader), warmup_epochs=5)
    scheduler_D_A = LR_Scheduler_Head(args.lr_scheduler, 0.0002,
                                  args.epochs, len(trainloader), warmup_epochs=5)

    for epoch in range(args.epochs):
        train_new_idea(trainloader, model, D_A, ce_loss, dice_loss, optimizer, optimizer_D_A, device, scheduler,
                       scheduler_D_A, epoch)

    # save weights
    save_files = {
        'backbone': model.backbone.state_dict(),
        'model': model.state_dict(),
    }
    torch.save(save_files,
               "/home/yongsen/Segments/new_idea/resnet152_{}.pth".format(
                   epoch))

def test(args):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # model = Student(num_classes=args.num_classes).to(device)
    model = GAN_Uet(num_classes=args.num_classes, vt_type=args.vit_name, img_size=args.img_size, middle=[9, 14, 19]).to(device)
    state = torch.load('/home/yongsen/Segments/new_idea/resnet152_399.pth', map_location=device)
    model_para = state['model']

    model.load_state_dict(model_para)

    db_test = HemangiomaDataset_val(r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/test/img',
                                 opt=transforms.Compose([
                                     transforms.Resize((args.img_size, args.img_size)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                 ])
                                 )

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)

    for sample in testloader:
        image_batch = sample['image']
        image_batch = image_batch.to(device)
        outputs, mid_outputs = model(image_batch)

        outputs = F.interpolate(outputs, sample['original_size'], mode='bilinear', align_corners=True)

        label = torch.argmax(outputs.squeeze(), axis=0)

        out = label.data.cpu().numpy()
        # Image.fromarray(((out > 0.5) * 255).astype(np.uint8)).save(os.path.join(save_path, name[0]))
        Image.fromarray(((out > 0.5) * 255).astype(np.uint8)).save(os.path.join(r'/mnt/disk/yongsen/Semantic_Segmentation/hemangioma/test/predict', sample['name'][0]))

        # one_hot = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        # print(one_hot)
        # label = torch.topk(one_hot, 1)[1].squeeze(1)
        # print(label)

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
    parser.add_argument('--pre_path', type=str,
                        default='/mnt/disk/yongsen/model/vit/imageNet21k/imagenet21k_ViT-L_16.npz')
    # parser.add_argument('--pre_path', type=str,
    #                     default='/mnt/disk/haoli/vit_models/imagenet21k/imagenet21k_ViT-B_16.npz')
    parser.add_argument('--cuda', default='cuda:0', help='disables CUDA training')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')

    parser.add_argument('--lr_scheduler', type=str, default='poly',
                        help='learning rate scheduler (default: poly)')
    parser.add_argument('--epochs', type=int,
                        default=250, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument('--workers', default=4, type=int)

    parser.add_argument('--base_lr', type=float, default=0.02,
                        help='segmentation network learning rate')
    parser.add_argument('--num_classes', type=int,
                        default=2, help='output channel of network')
    parser.add_argument('--img_size', type=int,
                        default=256, help='input patch size of network input')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='ViT-L_16', help='select one vit model', choices=['ViT-B_16, ViT-L_16'])
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')

    args = parser.parse_args()
    set_seed(args)
    main(args)
    test(args)