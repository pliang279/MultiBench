"""Implements decoders for robotics tasks."""
import torch
import torch.nn as nn
from .models_utils import init_weights
from .layers import (
    conv2d,
    predict_flow,
    deconv,
    crop_like
)


class OpticalFlowDecoder(nn.Module):
    """Implements optical flow and optical flow mask decoder."""
    
    def __init__(self, z_dim, initailize_weights=True):
        """Initialize OpticalFlowDecoder Module.

        Args:
            z_dim (float): Z dimension size
            alpha (float): Alpha to multiply proprio input by.
            initialize_weights (bool, optional): Whether to initialize weights or not. Defaults to True.
        """
        super().__init__()

        self.optical_flow_conv = conv2d(2 * z_dim, 64, kernel_size=1, stride=1)

        self.img_deconv6 = deconv(64, 64)
        self.img_deconv5 = deconv(64, 32)
        self.img_deconv4 = deconv(162, 32)
        self.img_deconv3 = deconv(98, 32)
        self.img_deconv2 = deconv(98, 32)

        self.predict_optical_flow6 = predict_flow(64)
        self.predict_optical_flow5 = predict_flow(162)
        self.predict_optical_flow4 = predict_flow(98)
        self.predict_optical_flow3 = predict_flow(98)
        self.predict_optical_flow2 = predict_flow(66)

        self.upsampled_optical_flow6_to_5 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow5_to_4 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow4_to_3 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow3_to_2 = nn.ConvTranspose2d(
            2, 2, 4, 2, 1, bias=False
        )

        self.predict_optical_flow2_mask = nn.Conv2d(
            66, 1, kernel_size=3, stride=1, padding=1, bias=False
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, tiled_feat, img_out_convs):
        """
        Predicts the optical flow and optical flow mask.

        Args:
            tiled_feat: action conditioned z (output of fusion + action network)
            img_out_convs: outputs of the image encoders (skip connections)
        """
        out_img_conv1, out_img_conv2, out_img_conv3, out_img_conv4, out_img_conv5, out_img_conv6 = (
            img_out_convs
        )

        optical_flow_in_f = torch.cat([out_img_conv6, tiled_feat], 1)
        optical_flow_in_f2 = self.optical_flow_conv(optical_flow_in_f)
        optical_flow_in_feat = self.img_deconv6(optical_flow_in_f2)

        # predict optical flow pyramids
        optical_flow6 = self.predict_optical_flow6(optical_flow_in_feat)
        optical_flow6_up = crop_like(
            self.upsampled_optical_flow6_to_5(optical_flow6), out_img_conv5
        )
        out_img_deconv5 = crop_like(
            self.img_deconv5(optical_flow_in_feat), out_img_conv5
        )

        concat5 = torch.cat(
            (out_img_conv5, out_img_deconv5, optical_flow6_up), 1)
        optical_flow5 = self.predict_optical_flow5(concat5)
        optical_flow5_up = crop_like(
            self.upsampled_optical_flow5_to_4(optical_flow5), out_img_conv4
        )
        out_img_deconv4 = crop_like(self.img_deconv4(concat5), out_img_conv4)

        concat4 = torch.cat(
            (out_img_conv4, out_img_deconv4, optical_flow5_up), 1)
        optical_flow4 = self.predict_optical_flow4(concat4)
        optical_flow4_up = crop_like(
            self.upsampled_optical_flow4_to_3(optical_flow4), out_img_conv3
        )
        out_img_deconv3 = crop_like(self.img_deconv3(concat4), out_img_conv3)

        concat3 = torch.cat(
            (out_img_conv3, out_img_deconv3, optical_flow4_up), 1)
        optical_flow3 = self.predict_optical_flow3(concat3)
        optical_flow3_up = crop_like(
            self.upsampled_optical_flow3_to_2(optical_flow3), out_img_conv2
        )
        out_img_deconv2 = crop_like(self.img_deconv2(concat3), out_img_conv2)

        concat2 = torch.cat(
            (out_img_conv2, out_img_deconv2, optical_flow3_up), 1)

        optical_flow2_unmasked = self.predict_optical_flow2(concat2)

        optical_flow2_mask = self.predict_optical_flow2_mask(concat2)

        optical_flow2 = optical_flow2_unmasked * \
            torch.sigmoid(optical_flow2_mask)

        return optical_flow2, optical_flow2_mask


class EeDeltaDecoder(nn.Module):
    """Implements an EE Delta Decoder."""
    
    def __init__(self, z_dim, action_dim, initailize_weights=True):
        """Initialize EeDeltaDecoder Module.

        Args:
            z_dim (float): Z dimension size
            alpha (float): Alpha to multiply proprio input by.
            initialize_weights (bool, optional): Whether to initialize weights or not. Defaults to True.
        """
        super().__init__()

        self.ee_delta_decoder = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, action_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, mm_act_feat):
        """Apply EeDeltaDecoder Module to EE Delta.

        Args:
            mm_act_feat (torch.Tensor): EE Delta

        Returns:
            torch.Tensor: Decoded Output
        """
        return self.ee_delta_decoder(mm_act_feat)


class ContactDecoder(nn.Module):
    """Decodes everything, given some input."""
    
    def __init__(self, z_dim, deterministic, head=1):
        """Initialize ContactDecoder Module.

        Args:
            z_dim (float): Z dimension size
            deterministic (float): Whether input parameters are deterministic or sampled from some distribution given prior mu and var.
            head (int): Output dimension of head.
        """
        super().__init__()

        self.deterministic = deterministic
        self.contact_fc = nn.Sequential(nn.Linear(z_dim, head))

    def forward(self, input):
        """Apply ContactDecoder Module to Layer Input.

        Args:
            input (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Decoded Output
        """
        if self.deterministic:
            z, mm_act_feat, tiled_feat, img_out_convs = input
        else:
            z, mm_act_feat, tiled_feat, img_out_convs, mu_z, var_z, mu_prior, var_prior = input

        contact_out = self.contact_fc(mm_act_feat)

        return contact_out
