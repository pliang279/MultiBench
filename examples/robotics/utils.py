import torch
import torch.nn as nn
import numpy as np
import random
import copy
import math
from tqdm import tqdm


def detach_var(var):
    """Detaches a var from torch

    Args:
        var (torch var): Torch variable that requires grad

    Returns:
        TYPE: numpy array
    """
    return var.cpu().detach().numpy()


def set_seeds(seed, use_cuda):
    """Set Seeds

    Args:
        seed (int): Sets the seed for numpy, torch and random
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if use_cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def quaternion_to_euler(x, y, z, w):

    # t0 = +2.0 * (w * x + y * z)
    # t1 = +1.0 - 2.0 * (x * x + y * y)
    # X = np.arctan2(t0, t1)

    # t2 = +2.0 * (w * y - z * x)
    # t2 = +1.0 if t2 > +1.0 else t2
    # t2 = -1.0 if t2 < -1.0 else t2
    # Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)

    Z = -Z - np.pi / 2

    return Z


def compute_accuracy(pred, target):
    pred_1 = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    target_1 = torch.where(target > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
    batch_size = target.size()[0] * 1.0

    num_correct = 1.0 * torch.where(
        pred_1 == target_1, torch.ones_like(pred), torch.zeros_like(pred)
    ).sum().float()

    accuracy = num_correct / batch_size
    return accuracy


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    image_transform = image * scale
    # torch.from_numpy(img.transpose((0, 3, 1, 2))).float()
    return image_transform.transpose(1, 3).transpose(2, 3)


def log_normal(x, m, v):

    log_prob = -((x - m) ** 2 / (2 * v)) - 0.5 * torch.log(2 * math.pi * v)

    return log_prob


def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = element_wise.sum(-1)
    return kl


def augment_val(val_filename_list, filename_list):

    filename_list1 = copy.deepcopy(filename_list)
    val_filename_list1 = []

    for name in tqdm(val_filename_list):
        filename = name[:-8]
        found = True

        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        if file_number < 10:
            comp_number = 19
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += -1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number < 0:
                    found = False
                    break
        else:
            comp_number = 0
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                filename1 not in val_filename_list1
            ):
                comp_number += 1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number > 19:
                    found = False
                    break

        if found:
            if filename1 in filename_list1:
                filename_list1.remove(filename1)

            if filename1 not in val_filename_list:
                val_filename_list1.append(filename1)

    val_filename_list1 += val_filename_list

    return val_filename_list1, filename_list1


def flow2rgb(flow_map, max_value=None):
    global args
    _, h, w = flow_map.shape
    # flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)


def scene_flow2rgb(flow_map):
    global args

    flow_map = np.where(flow_map > 1e-6, flow_map, np.zeros_like(flow_map))

    indices1 = np.nonzero(flow_map[0, :, :])
    indices2 = np.nonzero(flow_map[1, :, :])
    indices3 = np.nonzero(flow_map[2, :, :])

    normalized_flow_map = np.zeros_like(flow_map)

    divisor_1 = 0
    divisor_2 = 0
    divisor_3 = 0

    if np.array(indices1).size > 0:
        divisor_1 = (
            flow_map[0, :, :][indices1].max() - flow_map[0, :, :][indices1].min()
        )

    if np.array(indices2).size > 0:
        divisor_2 = (
            flow_map[1, :, :][indices2].max() - flow_map[1, :, :][indices2].min()
        )

    if np.array(indices3).size > 0:
        divisor_3 = (
            flow_map[2, :, :][indices3].max() - flow_map[2, :, :][indices3].min()
        )

    if divisor_1 > 0:
        normalized_flow_map[0, :, :][indices1] = (
            flow_map[0, :, :][indices1] - flow_map[0, :, :][indices1].min()
        ) / divisor_1

    if divisor_2 > 0:
        normalized_flow_map[1, :, :][indices2] = (
            flow_map[1, :, :][indices2] - flow_map[1, :, :][indices2].min()
        ) / divisor_2

    if divisor_3 > 0:
        normalized_flow_map[2, :, :][indices3] = (
            flow_map[2, :, :][indices3] - flow_map[2, :, :][indices3].min()
        ) / divisor_3

    return normalized_flow_map


def point_cloud2rgb(flow_map):
    global args

    flow_map = np.where(flow_map > 5e-4, flow_map, np.zeros_like(flow_map))

    flow_map = np.tile(
        np.expand_dims(np.sqrt(np.sum(np.square(flow_map), axis=0)), axis=0), (3, 1, 1)
    )
    return flow_map


def EPE(input_flow, target_flow, device, sparse=False, mean=True):
    # torch.cuda.init()

    EPE_map = torch.norm(target_flow.cpu() - input_flow.cpu(), 2, 1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

        EPE_map = EPE_map[~mask.data]
    if mean:
        return EPE_map.mean().to(device)
    else:
        return (EPE_map.sum() / batch_size).to(device)


def realEPE(output, target, device, sparse=False):
    b, _, h, w = target.size()

    upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
    return EPE(upsampled_output, target, device, sparse, mean=True)


def realAAE(output, target, device, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
    return AAE(upsampled_output, target, device, sparse, mean=True)


def AAE(input_flow, target_flow, device, sparse=False, mean=True):
    b, _, h, w = target_flow.size()
    ones = torch.ones([b, 1, h, w])
    target = torch.cat((target_flow.cpu(), ones), 1)
    inp = torch.cat((input_flow.cpu(), ones), 1)
    target = target.permute(0, 2, 3, 1).contiguous().view(b * h * w, -1)
    inp = inp.permute(0, 2, 3, 1).contiguous().view(b * h * w, -1)

    target = target.div(torch.norm(target, dim=1, keepdim=True).expand_as(target))
    inp = inp.div(torch.norm(inp, dim=1, keepdim=True).expand_as(inp))

    dot_prod = torch.bmm((target.view(b * h * w, 1, -1)), inp.view(b * h * w, -1, 1))
    AAE_map = torch.acos(torch.clamp(dot_prod, -1, 1))

    return AAE_map.mean().to(device)
