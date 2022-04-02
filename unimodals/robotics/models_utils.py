"""Utility functions for robotics unimodals."""
import torch
import torch.nn as nn


def init_weights(modules):
    """Weight initialization from original SensorFusion Code."""
    for m in modules:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def rescaleImage(image, output_size=128, scale=1 / 255.0):
    """Rescale the image in a sample to a given size.
    
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    image_transform = image * scale
    return image_transform.transpose(1, 3).transpose(2, 3)


def filter_depth(depth_image):
    """Get filter depth given a depth image.

    Args:
        depth_image (torch.Tensor): Depth image.

    Returns:
        torch.Tensor: Output
    """
    depth_image = torch.where(
        depth_image > 1e-7, depth_image, torch.zeros_like(depth_image)
    )
    return torch.where(depth_image < 2, depth_image, torch.zeros_like(depth_image))
