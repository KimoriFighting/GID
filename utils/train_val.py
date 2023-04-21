from tqdm import tqdm
import numpy as np
from medpy import metric
import torch
from scipy.ndimage import zoom
import torch.nn as nn
from torch.autograd import Variable

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

from utils.losses import draw_pit
def test_single_volume(image, label, net, classes, patch_size=[256, 256], testind=0):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs, _ = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                # draw_pit(out, f'{testind}_{ind}.png')
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()

    # from PIL import Image
    # for i in range(image.shape[0]):
    #     if testind>6:
    #         break
    #     if i < 70:
    #         continue
    #     # img_i1 = image[i, :, :]
    #     # img = Image.fromarray(np.uint8(img))
    #     # img_i1 = Image.fromarray(img_i1*255).convert('L')
    #     # img_i1.save(f'D:\Study\论文\picture\input\{testind}_{i}.png')
    #     draw_pit(prediction[i], f'/search/hadoop04/jiangxinfa/yongsen/TransUnet/GID_udf/{testind}_{i}.png')
    #     # draw_pit(label[i],  f'D:\Study\论文\picture\label\{testind}_{i}.png')

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    return metric_list

# from apex import amp
def train(trainloader, model, ce_loss, dice_loss, optimizer, device, scheduler, epoch):
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model.train()
    trainloader = tqdm(trainloader, ncols=100)
    mean_loss = torch.zeros(1).to(device)
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        scheduler(optimizer, i_batch, epoch)

        outputs, mid_outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss = 0.5*loss_ce + 0.5*loss_dice

        optimizer.zero_grad()
        #
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * i_batch + loss.detach()) / (i_batch + 1)
        trainloader.desc = "[epoch {}] mean_loss {} - loss_ce{} - loss_dice{}". \
            format(epoch, round(mean_loss.item(), 3), round(loss_ce.item(), 3),
                   round(loss_dice.item(), 3))

def train_new_idea(trainloader, generator, discriminator, ce_loss, dice_loss, optimizer_G, optimizer_D, device, scheduler_G, scheduler_D, epoch):
    Tensor = torch.cuda.FloatTensor #if cuda else torch.Tensor
    criterion_GAN = torch.nn.MSELoss()
    # criterion_identity = torch.nn.L1Loss()

    patch = (1, 256 // 2 ** 4, 256 // 2 ** 4)

    generator.train()
    discriminator.train()

    trainloader = tqdm(trainloader, ncols=130)

    mean_loss = torch.zeros(1).to(device)
    mean_ce = torch.zeros(1).to(device)
    mean_dice = torch.zeros(1).to(device)

    mean_gv = torch.zeros(1).to(device)
    mean_dv = torch.zeros(1).to(device)
    mean_df = torch.zeros(1).to(device)

    for i, sample in enumerate(trainloader):
        train_image, train_label = sample['image'], sample['label']
        train_image, train_label = train_image.to(device), train_label.to(device)

        # Adversarial ground truths [1, 1, 8, 8]
        valid = Variable(Tensor(np.ones((train_image.size(0), *patch))), requires_grad=False)  # [1, 1, 8, 8]
        fake = Variable(Tensor(np.zeros((train_image.size(0), *patch))), requires_grad=False)

        scheduler_G(optimizer_G, i, epoch)
        scheduler_D(optimizer_D, i, epoch)

        # train generator
        optimizer_G.zero_grad()

        outputs, mid = generator(train_image)
        label = torch.argmax(outputs, axis=1).unsqueeze(1)
        # indentity loss
        loss_GAN_BA = criterion_GAN(discriminator(train_image, label), valid)

        loss_ce = ce_loss(outputs, train_label[:].long())
        loss_dice = dice_loss(outputs, train_label, softmax=True)

        loss = loss_ce + loss_dice + loss_GAN_BA*0.1
        loss_tmp = 0.5*loss_ce + 0.5*loss_dice
        loss.backward()
        optimizer_G.step()

        # train Discriminator
        optimizer_D.zero_grad()
        loss_real = criterion_GAN(discriminator(train_image, train_label.unsqueeze(1)), valid)
        loss_fake = criterion_GAN(discriminator(train_image, label.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2
        loss_D.backward()
        optimizer_D.step()

        mean_loss = (mean_loss * i + loss_tmp.detach()) / (i + 1)
        mean_ce = (mean_ce * i + loss_ce.detach()) / (i + 1)
        mean_dice = (mean_dice * i + loss_dice.detach()) / (i + 1)
        mean_gv = (mean_gv * i + loss_GAN_BA.detach()) / (i + 1)
        mean_dv = (mean_dv * i + loss_real.detach()) / (i + 1)
        mean_df = (mean_df * i + loss_fake.detach()) / (i + 1)
        trainloader.desc = "[epoch {}] mean_loss {} loss_ce {} loss_dice {} loss_GAN_ga {} loss_dv {} loss_df {}". \
            format(epoch, round(mean_loss.item(), 3), round(mean_ce.item(), 3), round(mean_dice.item(), 3),
                    round(mean_gv.item(), 3), round(mean_dv.item(), 3), round(mean_df.item(), 3))

def train_stu(trainloader, model, teacher, ce_loss, dice_loss, kl_loss, avg_loss, optimizer, device, scheduler, epoch, args):
    model.train()
    # teacher.eval()
    trainloader = tqdm(trainloader, ncols=125)
    mean_loss = torch.zeros(1).to(device)
    meankl_loss = torch.zeros(1).to(device)
    meanavg_loss = torch.zeros(1).to(device)
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        scheduler(optimizer, i_batch, epoch)
        # pre [feature cb_23, cb_19, cb_14, cb_9]
        outputs, mid_outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss_l = 0.5*loss_ce + 0.5*loss_dice
        #
        t_outputs, mid_y = teacher(image_batch)
        # # #
        loss_kl = torch.zeros(1).to(device)
        # for i in range(args.n_skip):
        #     loss_kl += kl_loss(mid_y[0].detach(), mid_outputs[0])
        # #
        # if args.n_skip:
        #     loss_kl /= args.n_skip
        # # #
        # loss_avg = torch.zeros(1).to(device)
        loss_avg = avg_loss(outputs, t_outputs.detach())

        # loss = loss_l
        loss = loss_l + 0.01 * (loss_avg)
        # loss = loss_l + 0.01 * (loss_kl)
        # loss = loss_l + 0.1 * (loss_kl) + 0.1 * loss_avg
        # loss = loss_l + 1/(epoch+1) * 0.1 * (loss_kl)  # 1/x
        # loss = loss_l + np.sqrt(1-np.power(epoch/args.epochs, 2) + 1e-6) * 0.1 * (loss_kl)  # x2+y2 = 1
        # loss = loss_l + (args.epochs-epoch)/args.epochs * 0.1 * (loss_kl) # x+y=1
        # if epoch % 10 == 0 and epoch <= 300:
        #     loss = loss_l + 0.1 * (loss_kl) * np.sqrt(1-np.power(epoch/args.epochs, 2) + 1e-6)
        # else:
        #     loss = loss_l
        optimizer.zero_grad()

        loss.backward()
        #
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)

        optimizer.step()
        mean_loss = (mean_loss * i_batch + loss_l.detach()) / (i_batch + 1)
        # meankl_loss = torch.zeros(1)
        # meanavg_loss = torch.zeros(1)
        meankl_loss = (meankl_loss * i_batch + loss_kl.detach()) / (i_batch + 1)
        meanavg_loss = (meanavg_loss * i_batch + loss_avg.detach()) / (i_batch + 1)
        trainloader.desc = "[epoch {}] mean_loss {} - loss_ce{} - loss_dice{} - loss_kl{} - loss_avg{}". \
            format(epoch, round(mean_loss.item(), 3), round(loss_ce.item(), 3),
                   round(loss_dice.item(), 3), round(meankl_loss.item(), 3), round(meanavg_loss.item(), 3))
    return mean_loss.item(), meanavg_loss.item()

def train_smi_stu(trainloader, model, teacher, ce_loss, dice_loss, kl_loss, l1_loss, optimizer, device, scheduler, epoch):
    model.train()
    # teacher.eval()
    trainloader = tqdm(trainloader, ncols=125)
    mean_loss = torch.zeros(1).to(device)
    for i_batch, sampled_batch in enumerate(trainloader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.to(device), label_batch.to(device)
        scheduler(optimizer, i_batch, epoch)
        # pre [feature cb_23, cb_19, cb_14, cb_9]
        outputs, mid_outputs = model(image_batch)

        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss_l = 0.5*loss_ce + 0.5*loss_dice

        feature, mid_y = teacher(image_batch, 1)

        # loss_kl = 0.0
        loss_kl = kl_loss(feature.detach(), mid_outputs[0])
        # # loss_kl = kl_loss(mid_y[0].detach(), mid_outputs[1])
        # loss_l1 = torch.tensor(0).to(device)
        # for i in range(len(mid_y)):
        #     loss_l1 += kl_loss(mid_y[i].detach(), mid_outputs[i+1])

        # loss_l1 /= len(mid_y)

        # loss = loss_l + 0.1*(loss_kl) + 0.1*loss_l1
        loss = loss_l + 0.1 * (loss_kl)
        # loss = loss_l + 1/(epoch+1) * 0.1 * (loss_kl)  # 1/x
        # loss = loss_l + np.sqrt(1-np.power(epoch/200, 2) + 1e-6) * 0.1 * (loss_kl)  # x2+y2 = 1
        # loss = loss_l + (200-epoch)/200 * 0.1 * (loss_kl) # x+y=1
        # if epoch < 100:
        #     loss = loss_l + 0.1 * (loss_kl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss = (mean_loss * i_batch + loss_l.detach()) / (i_batch + 1)
        trainloader.desc = "[epoch {}] mean_loss {} - loss_ce {} - loss_dice {} - loss_kl {}". \
            format(epoch, round(mean_loss.item(), 3), round(loss_ce.item(), 3),
                   round(loss_dice.item(), 3), round(loss_kl.item(), 3))
        # trainloader.desc = "[epoch {}] mean_loss {} - loss_ce {} - loss_dice {} - loss_kl {}- loss_l1 {}". \
        #     format(epoch, round(mean_loss.item(), 3), round(loss_ce.item(), 3),
        #            round(loss_dice.item(), 3), round(loss_kl.item(), 3),  round(loss_l1.item(), 3))
    return mean_loss.item()

def val(args, testloader, model, len):
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size], testind=i_batch)
        metric_list += np.array(metric_i)
        print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len
    for i in range(1, args.num_classes):
        print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

from medpy.metric.binary import hd, dc
def metrics(img_gt, img_pred):
    """
    Function to compute the metrics between two segmentation maps given as input.

    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.

    img_pred: np.array
    Array of the predicted segmentation map.

    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.

    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """

    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))

    res = []
    # Loop on each classes of the input images
    for c in [3, 1, 2]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0

        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        dice = dc(gt_c_i, pred_c_i)

        # Compute volume
        # volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.
        # volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice]

    return res

def val_acdc(model, valloader, device):
    dice = 0
    dice_lv = 0
    dice_rv = 0
    dice_myo = 0

    metric_list = 0.0
    valloader = tqdm(valloader)
    for step, sampled in enumerate(valloader):
        h, w = sampled["image"].size()[2:]
        image, label = sampled["image"], sampled["label"]
        image = image.to(device)
        model.eval()
        with torch.no_grad():
            outputs, _ = model(image)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            out = out.cpu().numpy()
            label = label.cpu().numpy()
            ans = metrics(label, out)

            dice_lv += ans[0]
            dice_rv += ans[1]
            dice_myo += ans[2]
            dice = (dice_lv + dice_rv + dice_myo) / 3

            valloader.desc = "[epoch:{}] dice:{} - dice_lv:{} - dice_rv:{} - dice_myo:{}". \
                format(1, round(dice / (step + 1), 4), round(dice_lv / (step + 1), 4),
                       round(dice_rv / (step + 1), 4), round(dice_myo / (step + 1), 4))
            # print(ans)
        # metric_list += np.array(metric_i)
        # tmp = metric_list/len
        # performance = np.mean(tmp, axis=0)[0]
        # mean_hd95 = np.mean(tmp, axis=0)[1]
        # valloader.desc = "[epoch:{}] performance:{} - mean_hd95:{}". \
        #     format(epoch, round(performance.item(), 3), round(mean_hd95.item(), 3))
