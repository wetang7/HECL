# IDRiD image size: (4288, 2848)
import warnings
from xml.sax.xmlreader import InputSource
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from network import NestedUNet
from Unet2d import UNet2D
from cenet import CE_Net_
from dataset import OrganData
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='dr_NestedUNet', help='model_name')

# epoch, batch_size setting
parser.add_argument('--batch_size', type=int, default=6, help='batch_size per gpu')
parser.add_argument('--seed', type=int,  default=43, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')

# training/testing sets
parser.add_argument("--data_path", type=str, default='DR_Segmentation/IDRiD/Hard Exudates')

# learning rate, optimizer setting
parser.add_argument("--deterministic", type=bool, default=True)
parser.add_argument("--patch_num", type=int, default=16)

args = parser.parse_args()

snapshot_path = "DR_Segmentation/evaluation/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

batch_size = args.batch_size

n_channels = 3
n_labels = 1

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def compute_performance(binary_output, label, metric, prefix=None, reduction='mean'):

    binary_output = torch.from_numpy(binary_output)
    label = torch.from_numpy(label)

    result = {}

    if prefix:
        result = {prefix+'_'+key:value for key, value in result.items()}

    return result


def test(checkpoint, eval_loader):

    net = NestedUNet()

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
    print(masks_pred.shape)
    masks_gt = np.array(masks_gt)   #.transpose((1, 0, 2, 3))
    masks_pred = np.where(masks_pred > 0.5, 1, 0)
    masks_pred_raw = masks_pred*255
    masks_gt_raw = np.where(masks_gt > 0, 255, 0)

    metrics = compute_performance(masks_pred, masks_gt,
                                  metric=['confusion','IoU'],
                                  prefix='test', reduction='none')

    recall = 0
    precision = 0
    f1 = 0
    IoU = 0
    acc = 0

    return IoU, f1, recall, precision, acc, masks_gt_raw, masks_pred_raw


   
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
    
    net = NestedUNet()

    with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:

        test_dataset = OrganData(data_path=os.path.join(args.data_path,'test'), size=(4288, 2848), transform=False)
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last = False)
        
        path_checkpoint = "/home/wetang7/DR_Segmentation/output/dr_NestedUNet_contr_eg_pt_0.01/model"
        checkpoint = torch.load(path_checkpoint)

        IoU, f1, recall, precision, acc, masks_gt_raw, masks_pred_raw = test(checkpoint, eval_loader)
        masks_pred_raw = masks_pred_raw[:, 0, ...]
        ## save model
        for i in range(27):
            save_image_path = os.path.join(snapshot_path + '/pred_image', 'img_' + str(55+i) + '.png')
            print(masks_pred_raw.shape)
            pred_image = Image.fromarray(np.array(masks_pred_raw[i], dtype='uint8'))
            pred_image.save(save_image_path)
