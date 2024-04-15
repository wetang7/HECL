import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import numpy as np
from torch.autograd import Variable


import os

#seed = int(os.environ.get("PL_GLOBAL_SEED"))
#pl.utilities.seed.seed_everything(seed=seed)

class SyncFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class FocalLoss(nn.Module):
    """
    This class implements the segmentation focal loss.
    https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        """
        Constructor method
        :param alpha: (float) Alpha constant
        :param gamma: (float) Gamma constant (see paper)
        """
        # Call super constructor
        super(FocalLoss, self).__init__()
        # Save parameters
        self.alpha = alpha
        self.gamma = gamma

    def __repr__(self):
        """
        Get representation of the loss module
        :return: (str) String including information
        """
        return "{}, alpha={}, gamma={}".format(self.__class__.__name__, self.alpha, self.gamma)

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computes the binary cross entropy loss of segmentation masks
        :param prediction: (torch.Tensor) Prediction probability
        :param label: (torch.Tensor) Label one-hot encoded
        :return: (torch.Tensor) Loss value
        """
        # Calc binary cross entropy loss as normal
        prediction = torch.sigmoid(prediction)
        binary_cross_entropy_loss = -(label * torch.log(prediction.clamp(min=1e-12))
                                      + (1.0 - label) * torch.log((1.0 - prediction).clamp(min=1e-12)))
        # Calc focal loss factor based on the label and the prediction
        focal_factor = prediction * label + (1.0 - prediction) * (1.0 - label)
        # Calc final focal loss
        loss = ((1.0 - focal_factor) ** self.gamma * binary_cross_entropy_loss * self.alpha).sum(dim=1).mean()
        return loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5

    loss = 0
    for i in range(target.shape[1]):
        intersect = torch.sum(score[:, i, ...] * target[:, i, ...])
        z_sum = torch.sum(score[:, i, ...] )
        y_sum = torch.sum(target[:, i, ...] )
        loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss * 1.0 / target.shape[1]

    return loss
