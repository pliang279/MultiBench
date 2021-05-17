import torch.nn as nn
from .models_utils import filter_depth, init_weights, rescaleImage
from .layers import CausalConv1D, Flatten, conv2d


class ProprioEncoder(nn.Module):
    def __init__(self, z_dim, alpha, initialize_weights=True):
        """
        Image encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.proprio_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 2 * self.z_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, proprio, training=False):
        return self.proprio_encoder(self.alpha * proprio).unsqueeze(2)


class ForceEncoder(nn.Module):
    def __init__(self, z_dim, alpha, initialize_weights=True):
        """
        Force encoder taken from selfsupervised code
        """
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.frc_encoder = nn.Sequential(
            CausalConv1D(6, 16, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(16, 32, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(32, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(64, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, 2 * self.z_dim, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, force, training=False):
        return self.frc_encoder(self.alpha * force)


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, alpha, initialize_weights=True):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.img_conv1 = conv2d(3, 16, kernel_size=7, stride=2)
        self.img_conv2 = conv2d(16, 32, kernel_size=5, stride=2)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2)
        self.img_conv4 = conv2d(64, 64, stride=2)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2)
        self.img_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, vis_in, training=False):
        image = rescaleImage(vis_in)

        # image encoding layers
        out_img_conv1 = self.img_conv1(self.alpha * image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)

        img_out_convs = (
            out_img_conv1,
            out_img_conv2,
            out_img_conv3,
            out_img_conv4,
            out_img_conv5,
            out_img_conv6,
        )

        # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        img_out = self.img_encoder(flattened).unsqueeze(2)

        return img_out, img_out_convs


class DepthEncoder(nn.Module):
    def __init__(self, z_dim, alpha, initialize_weights=True):
        """
        Simplified Depth Encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim
        self.alpha = alpha

        self.depth_conv1 = conv2d(1, 32, kernel_size=3, stride=2)
        self.depth_conv2 = conv2d(32, 64, kernel_size=3, stride=2)
        self.depth_conv3 = conv2d(64, 64, kernel_size=4, stride=2)
        self.depth_conv4 = conv2d(64, 64, stride=2)
        self.depth_conv5 = conv2d(64, 128, stride=2)
        self.depth_conv6 = conv2d(128, self.z_dim, stride=2)

        self.depth_encoder = nn.Linear(4 * self.z_dim, 2 * self.z_dim)
        self.flatten = Flatten()

        if initialize_weights:
            init_weights(self.modules())

    def forward(self, depth_in, training=False):
        depth = filter_depth(depth_in)

        # depth encoding layers
        out_depth_conv1 = self.depth_conv1(self.alpha * depth)
        out_depth_conv2 = self.depth_conv2(out_depth_conv1)
        out_depth_conv3 = self.depth_conv3(out_depth_conv2)
        out_depth_conv4 = self.depth_conv4(out_depth_conv3)
        out_depth_conv5 = self.depth_conv5(out_depth_conv4)
        out_depth_conv6 = self.depth_conv6(out_depth_conv5)

        depth_out_convs = (
            out_depth_conv1,
            out_depth_conv2,
            out_depth_conv3,
            out_depth_conv4,
            out_depth_conv5,
            out_depth_conv6,
        )

        # depth embedding parameters
        flattened = self.flatten(out_depth_conv6)
        depth_out = self.depth_encoder(flattened).unsqueeze(2)

        return depth_out, depth_out_convs


class ActionEncoder(nn.Module):
    def __init__(self, action_dim):
        """
        Action Encoder
        """
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, action, training=False):
        if action is None:
            return None
        return self.action_encoder(action)
