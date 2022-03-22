"""Implements sensor fusion networks for supervised and self-supervised objectives."""
import torch
import torch.nn as nn
from .models_utils import (
    duplicate,
    gaussian_parameters,
    product_of_experts,
    sample_gaussian,
)


class SensorFusion(nn.Module):
    """
    Implements traditional SensorFusionNetwork.
    
    #
        Regular SensorFusionNetwork Architecture
        Number of parameters:
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """
    
    def __init__(
        self, device, z_dim=128, action_dim=4, encoder=False, deterministic=False
    ):
        """Initialize SensorFusionNetwork.

        Args:
            device (torch.device): Device to train/test model on.
            z_dim (int, optional): Z dimension size. Defaults to 128.
            action_dim (int, optional): Action dimension size. Defaults to 4.
            encoder (bool, optional): Whether to apply action encoder or not. Defaults to False.
            deterministic (bool, optional): Whether the fusion networks is deterministic or not. Defaults to False.
        """
        super().__init__()

        self.z_dim = z_dim
        self.encoder_bool = encoder
        self.device = device
        self.deterministic = deterministic

        # zero centered, 1 std normal distribution
        self.z_prior_m = torch.nn.Parameter(
            torch.zeros(1, self.z_dim), requires_grad=False
        )
        self.z_prior_v = torch.nn.Parameter(
            torch.ones(1, self.z_dim), requires_grad=False
        )
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        # -----------------------
        # action fusion network
        # -----------------------
        self.st_fusion_fc1 = nn.Sequential(
            nn.Linear(32 + self.z_dim, 128), nn.LeakyReLU(0.1, inplace=True)
        )
        self.st_fusion_fc2 = nn.Sequential(
            nn.Linear(128, self.z_dim), nn.LeakyReLU(0.1, inplace=True)
        )

        if deterministic:
            # -----------------------
            # modality fusion network
            # -----------------------
            # 4 Total modalities each (2 * z_dim)
            self.fusion_fc1 = nn.Sequential(
                nn.Linear(4 * 2 * self.z_dim,
                          128), nn.LeakyReLU(0.1, inplace=True)
            )
            self.fusion_fc2 = nn.Sequential(
                nn.Linear(self.z_dim, self.z_dim), nn.LeakyReLU(
                    0.1, inplace=True)
            )

        # -----------------------
        # weight initialization
        # -----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward_encoder(self, img_encoded, frc_encoded, proprio_encoded, depth_encoded, action_encoded):
        """Encode output using forward encoder.

        Args:
            img_encoded (torch.Tensor): Encoded image.
            frc_encoded (torch.Tesnor): Encoded force.
            proprio_encoded (torch.Tensor): Encoded proprioception
            depth_encoded (torch.Tensor): Encoded depth.
            action_encoded (torch.Tensor): Encoded action.

        Returns:
            torch.Tensor: tuple of outputs
        """
        # Get encoded outputs
        img_out, img_out_convs = img_encoded
        depth_out, depth_out_convs = depth_encoded
        frc_out = frc_encoded
        proprio_out = proprio_encoded

        # batch size
        batch_dim = img_out.size()[0]

        if self.deterministic:
            # multimodal embedding
            mm_f1 = torch.cat(
                [img_out, frc_out, proprio_out, depth_out], 1).squeeze()
            mm_f2 = self.fusion_fc1(mm_f1)
            z = self.fusion_fc2(mm_f2)

        else:
            # Encoder priors
            mu_prior, var_prior = self.z_prior

            # Duplicate prior parameters for each data point in the batch
            mu_prior_resized = duplicate(mu_prior, batch_dim).unsqueeze(2)
            var_prior_resized = duplicate(var_prior, batch_dim).unsqueeze(2)

            # Modality Mean and Variances
            mu_z_img, var_z_img = gaussian_parameters(img_out, dim=1)
            mu_z_frc, var_z_frc = gaussian_parameters(frc_out, dim=1)
            mu_z_proprio, var_z_proprio = gaussian_parameters(
                proprio_out, dim=1)
            mu_z_depth, var_z_depth = gaussian_parameters(depth_out, dim=1)

            # Tile distribution parameters using concatonation
            m_vect = torch.cat(
                [mu_z_img, mu_z_frc, mu_z_proprio,
                    mu_z_depth, mu_prior_resized], dim=2
            )
            var_vect = torch.cat(
                [var_z_img, var_z_frc, var_z_proprio,
                    var_z_depth, var_prior_resized],
                dim=2,
            )

            # Fuse modalities mean / variances using product of experts
            mu_z, var_z = product_of_experts(m_vect, var_vect)

            # Sample Gaussian to get latent
            z = sample_gaussian(mu_z, var_z, self.device)

        if self.encoder_bool or action_encoded is None:
            if self.deterministic:
                return img_out, frc_out, proprio_out, depth_out, z
            else:
                return img_out_convs, img_out, frc_out, proprio_out, depth_out, z
        else:
            # action embedding
            act_feat = action_encoded

            # state-action feature
            mm_act_f1 = torch.cat([z, act_feat], 1)
            mm_act_f2 = self.st_fusion_fc1(mm_act_f1)
            mm_act_feat = self.st_fusion_fc2(mm_act_f2)

            if self.deterministic:
                return img_out_convs, mm_act_feat, z
            else:
                return img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior

    def weight_parameters(self):
        """Get weight parameters."""
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        """Get bias parameters."""
        return [param for name, param in self.named_parameters() if "bias" in name]


class SensorFusionSelfSupervised(SensorFusion):
    """
    Implements SensorFusionNetwork for Self-Supervision.
    
        Regular SensorFusionNetwork Architecture
        Inputs:
            image:   batch_size x 3 x 128 x 128
            force:   batch_size x 6 x 32
            proprio: batch_size x 8
            action:  batch_size x action_dim
    """

    def __init__(
        self, device, z_dim=128, encoder=False, deterministic=False
    ):
        """Initialize SensorFusionSelfSupervised Module.

        Args:
            device (torch.Device): Device to train/test on.
            z_dim (int, optional): Z dimension size for encoders. Defaults to 128.
            encoder (bool, optional): Whether to apply the encoders or not. Defaults to False.
            deterministic (bool, optional): Whether the fusion network is deterministic or not. Defaults to False.
        """
        super().__init__(device, z_dim, encoder, deterministic)

        self.deterministic = deterministic

    def forward(self, input):
        """Apply SensorFusionSelfSupervised Module to Layer Input.

        Args:
            input (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        img_encoded, frc_encoded, proprio_encoded, depth_encoded, action_encoded = input

        if self.encoder_bool:
            # returning latent space representation if model is set in encoder mode
            z = self.forward_encoder(
                img_encoded, frc_encoded, proprio_encoded, depth_encoded, action_encoded)
            return z

        elif action_encoded is None:
            z = self.forward_encoder(
                img_encoded, frc_encoded, proprio_encoded, depth_encoded, None)
            pair_out = self.pair_fc(z)
            return pair_out

        else:
            if self.deterministic:
                img_out_convs, mm_act_feat, z = self.forward_encoder(
                    img_encoded, frc_encoded, proprio_encoded, depth_encoded, action_encoded
                )
            else:
                img_out_convs, mm_act_feat, z, mu_z, var_z, mu_prior, var_prior = self.forward_encoder(
                    img_encoded,
                    frc_encoded,
                    proprio_encoded,
                    depth_encoded,
                    action_encoded,
                )

        # ---------------- Training Objectives ----------------

        # tile state-action features and append to conv map
        batch_dim = mm_act_feat.size(0)  # batch size
        tiled_feat = mm_act_feat.view(
            batch_dim, self.z_dim, 1, 1).expand(-1, -1, 2, 2)

        if self.deterministic:
            return z, mm_act_feat, tiled_feat, img_out_convs
        else:
            return z, mm_act_feat, tiled_feat, img_out_convs, mu_z, var_z, mu_prior, var_prior


class roboticsConcat(nn.Module):
    """Concatenates tensors for robotics fusion."""
    
    def __init__(self, name=None):
        """Initialize roboticsConcat module.

        Args:
            name (str, optional): What kind of concatenation to do. Can be "noconcat", "image" or "simple". Defaults to None.
        """
        super(roboticsConcat, self).__init__()
        self.name = name

    def forward(self, x):
        """Apply roboticsConcat module to Layer Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer output.
        """
        if self.name == "noconcat":
            return [x[0][0].squeeze(), x[1].squeeze(), x[2].squeeze(), x[3][0].squeeze(), x[4]]
        if self.name == "image":
            return torch.cat([x[0][0].squeeze(), x[1][0].squeeze(), x[2]], 1)
        if self.name == "simple":
            return torch.cat([x[0].squeeze(), x[1]], 1)
        return torch.cat([x[0][0].squeeze(), x[1].squeeze(), x[2].squeeze(), x[3][0].squeeze(), x[4]], 1)
