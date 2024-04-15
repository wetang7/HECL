
# IDRiD image size: (4288, 2848)

import os
import sys
from tqdm import tqdm

import argparse
import logging
import time
import numpy as np
import torch
from cenet import CE_Net_
from utils.losses import BCELoss
from dataset import OrganData, get_coutour_embeddings, get_background_embeddings

from monai import metrics
from PIL import Image

from torch.utils.data import DataLoader
from pytorch_metric_learning import losses
from utils.metrics import compute_performance


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='dr_contr_CE_Net', help='model_name')

# epoch, batch_size setting
parser.add_argument('--max_epoch', type=int,  default=300, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

# training/testing sets
parser.add_argument("--data_path", type=str, default='DR_Segmentation/IDRiD/Hard Exudates')

# learning rate, optimizer setting
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--contr_loss_weight", type=float, default=0.1)
parser.add_argument('--display_freq', type=int, default=30, help='writer frequency for tensorboard')
parser.add_argument("--patch_num", type=int, default=5)
parser.add_argument("--bdp_threshold", type=int, default=0.3)
parser.add_argument("--fdp_threshold", type=int, default=0.3)

args = parser.parse_args()

snapshot_path = "DR_Segmentation/output/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

batch_size = args.batch_size
base_lr = args.lr
max_epoch = args.max_epoch
display_freq = args.display_freq
patch_num = args.patch_num
bdp_threshold = args.bdp_threshold
fdp_threshold = args.fdp_threshold
RESUME = False

def test(checkpoint, eval_loader):

    net = CE_Net_()
    model = net.to(device)
    model.load_state_dict(checkpoint)

    model.eval()
    masks_pred = []
    masks_gt = []

    with torch.set_grad_enabled(False):
        for _, sampled_batch in enumerate(eval_loader):
            inputs, gts = sampled_batch['image'], sampled_batch['label']
            inputs = inputs.to(device=device, dtype=torch.float)
            gts = gts.to(device=device, dtype=torch.float)

            pred_batch, feature = model(inputs)
            pred_sigmoid_batch = torch.sigmoid(pred_batch).cpu().numpy()
            gt_batch = gts[:, 0, ...].unsqueeze(1).cpu().numpy()

            masks_pred.extend(pred_sigmoid_batch)
            masks_gt.extend(gt_batch)

    masks_pred = np.array(masks_pred)   #.transpose((1, 0, 2, 3))
    masks_score = masks_pred
    print("masks_pred: ", masks_pred.shape)
    masks_gt = np.array(masks_gt)   #.transpose((1, 0, 2, 3))
    print("masks_gt: ", masks_gt.shape)
    masks_pred = np.where(masks_pred > 0.5, 1, 0)
    masks_pred_raw = masks_pred*255
    masks_gt_raw = np.where(masks_gt > 0, 255, 0)

    metrics = compute_performance(masks_score, masks_pred, masks_gt,
                                  metric=['confusion','IoU','AUC'],
                                  prefix='test', reduction='none')

    recall = np.array(metrics['test_recall'].cpu()).mean()
    precision = np.array(metrics['test_precision'].cpu()).mean()
    f1 = np.array(metrics['test_f1'].cpu()).mean()
    IoU = np.array(metrics['test_IoU'].cpu()).mean()    
    auc = metrics['test_AUC']
    acc = np.array(metrics['test_acc'].cpu()).mean()

    return IoU, auc, f1, recall, precision, acc, masks_gt_raw[0,:,:,:], masks_pred_raw[0,:,:,:]


   
if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')
    if not os.path.exists(snapshot_path + '/pred_image'):
        os.makedirs(snapshot_path + '/pred_image')


    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # define dataset, model, optimizer
    
    net = CE_Net_()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)      # subject to change
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, verbose=False)

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)
    dataset = OrganData(data_path=os.path.join(args.data_path,'train'), size=(4288, 2848), transform=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if RESUME:
        path_checkpoint = "DR_Segmentation/output/dr_NestedUNet_contr/model"
        checkpoint = torch.load(path_checkpoint)

        net.load_state_dict(checkpoint) 

    for epoch_num in tqdm(range(max_epoch), ncols=70):
                # epoch_num = epoch_num+100
        print("\n************* epoch %d begins *************" % epoch_num)

        if epoch_num <= 100:
            lr=3e-4
        elif epoch_num <= 200:
            lr=0.5*3e-4
        elif epoch_num <= 300:
            lr=0.25*3e-4
            
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)

        net.train()
        time1 = time.time()
        iter_num = 0


        for i_batch, sampled_batch in enumerate(dataloader):
            time2 = time.time()
            # obtain training data
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            label_batch = label_batch[:, 0, ...].unsqueeze(1)

            net = net.cuda()
            volume_batch = volume_batch.cuda()
            label_batch = label_batch.cuda()

            # get embeddings and outputs
            output, embedding = net(volume_batch)
            
            sup_loss = BCELoss()
            sup_loss = sup_loss(output, label_batch.float())

            b, c = embedding.size(0), embedding.size(1)
            h, w = embedding.size(2) // patch_num, embedding.size(3) // patch_num

            contour_label_patches = []
            bg_label_patches = []

            fdp_embeddings = []
            bdp_embeddings = []

            label_bg_in_batch = torch.empty(label_batch.shape)
            label_contour_in_batch = torch.empty(label_batch.shape)

            fdp_fc = []
            bdp_fc = []

            for ii in range(b):

                contour_label_patches = []
                bg_label_patches = []

                for i in range(patch_num * patch_num):
                    j = i // patch_num
                    k = i % patch_num
                    patch_i = embedding[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]
                    label_i = label_batch[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w]

                    fc = torch.sum(label_i) / (label_i.size(1)*label_i.size(2))

                    if fc == 0:
                        all_bg = label_i
                        contour_label_patches.append(all_bg)
                        bg_label_patches.append(all_bg)
                        bdp_embeddings.append((torch.sum(patch_i, (-1,-2))/(h*w)).unsqueeze(0))
                        bdp_fc.append(fc)

                    elif fc == 1:
                        all_contour = torch.add(label_i, -1)
                        contour_label_patches.append(all_contour)
                        bg_label_patches.append(all_contour)
                        fdp_embeddings.append((torch.sum(patch_i, (-1,-2))/(h*w)).unsqueeze(0))
                        fdp_fc.append(fc)

                    elif fc >= fdp_threshold:
                        contour_embeddings, contour = get_coutour_embeddings(label_i.cpu(), patch_i, iteration = 2)
                        bg_embeddings, bg = get_background_embeddings(label_i.cpu(), patch_i, iteration = 2)

                        contour_label_patches.append(contour)
                        bg_label_patches.append(bg)

                        fdp_embeddings.append((torch.sum(patch_i, (-1,-2))/(h*w)).unsqueeze(0))
                        fdp_fc.append(fc)

                    else: 
                        contour_embeddings, contour = get_coutour_embeddings(label_i.cpu(), patch_i, iteration = 1)
                        bg_embeddings, bg = get_background_embeddings(label_i.cpu(), patch_i, iteration = 5)

                        contour_label_patches.append(contour)
                        bg_label_patches.append(bg)

                        bdp_embeddings.append((torch.sum(patch_i, (-1,-2))/(h*w)).unsqueeze(0))
                        bdp_fc.append(fc)
                    
                    j = i // patch_num
                    k = i % patch_num
                    label_bg_in_batch[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w] = bg_label_patches[i]
                    label_contour_in_batch[ii, :, j * h: (j + 1) * h, k * w: (k + 1) * w] = contour_label_patches[i]

            embedding = embedding.cpu()
            contour_ems = torch.mul(label_contour_in_batch, embedding)
            bg_ems = torch.mul(label_bg_in_batch, embedding)
            contour_ems = torch.sum(contour_ems, (-1,-2)).cuda()/torch.sum(label_contour_in_batch, (-1,-2)).cuda()
            bg_ems = torch.sum(bg_ems, (-1,-2)).cuda()/torch.sum(label_bg_in_batch, (-1,-2)).cuda()

            ct_bg_em = torch.cat((contour_ems, bg_ems), 0)
            eg_labels = np.concatenate([np.ones(contour_ems.shape[0]), np.zeros(bg_ems.shape[0])])
            eg_labels = torch.from_numpy(eg_labels)
            eg_contr_loss = cont_loss_func(ct_bg_em, eg_labels)

            sorted_fdp_fc_id = sorted(range(len(fdp_fc)), key=lambda k: fdp_fc[k], reverse=True)
            sorted_bdp_fc_id = sorted(range(len(bdp_fc)), key=lambda k: bdp_fc[k], reverse=False)

            bdp_patches = torch.cat([bdp_embeddings[i] for i in sorted_bdp_fc_id[0:3]], 0)
            fdp_patches = torch.cat([fdp_embeddings[i] for i in sorted_fdp_fc_id[0:3]], 0)

            fdp_bdp_em = torch.cat((fdp_patches, bdp_patches), 0)
            pt_labels = np.concatenate([np.ones(fdp_patches.shape[0]), np.zeros(bdp_patches.shape[0])])
            pt_labels = torch.from_numpy(pt_labels)
            pt_contr_loss = cont_loss_func(fdp_bdp_em, pt_labels).cuda()

            total_loss = sup_loss + 0.01*(eg_contr_loss + pt_contr_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

        ## evaluation
        with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:

            print("epoch {} testing".format(epoch_num), file=f)
            test_dataset = OrganData(data_path=os.path.join(args.data_path,'test'), size=(4288, 2848), transform=False)
            eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last = False)

            IoU, auc, f1, recall, precision, acc, masks_gt_raw, masks_pred_raw = test(net.state_dict(), eval_loader)
            print(f"IoU: {IoU:1.4f} - auc: {auc:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f}", file=f)

        ## save model
        save_mode_path = os.path.join(snapshot_path + '/model', 'epoch_' + str(epoch_num) + '.pth')
        save_image_path = os.path.join(snapshot_path + '/pred_image', 'epoch_' + str(epoch_num) + '.png')
        if epoch_num % display_freq == 0:
            pred_image = Image.fromarray(np.array(masks_pred_raw[0], dtype='uint8'))
            pred_image.save(save_image_path)
            torch.save(net.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

