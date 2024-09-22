import os
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from safetensors.torch import save_file, load_file
from .distributions import DiagonalGaussianDistribution
from .discriminator import NLayerDiscriminator

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.layers = nn.ModuleList()
        if with_conv:
            self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=2, padding=1))
        else:
            self.layers.append(nn.AvgPool2d((2, 2)))

    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.layers:
            h = module(h)
        return h

class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if with_conv:
            self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1))

    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.layers:
            h = module(h)
        return h

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, act, conv_shortcut=False, dropout=0.):
        super().__init__()
        self.act = act
        self.block1 = nn.ModuleList()
        self.block1.append(nn.Sequential(
            nn.BatchNorm2d(in_features),
            self.act,
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        ))
        self.block2 = nn.ModuleList()
        self.block2.append(nn.Sequential(
            nn.BatchNorm2d(out_features),
            self.act,
            nn.Dropout(dropout),
            nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        ))
        
        self.layers = None
        if in_features!= out_features:
            self.layers = nn.ModuleList()
            if conv_shortcut:
                self.layers.append(nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1)) # x to x
            else:
                self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(1, 1), stride=1, padding=0)) # x to x
    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.block1:
            h = module(h)
        
        for module in self.block2:
            h = module(h)

        if self.layers != None:
            for module in self.layers:
                x = module(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.norm = nn.GroupNorm(num_groups=32, num_channels=features, eps=1e-6, affine=True)

        self.q_proj = nn.Conv2d(features, features, kernel_size=(1, 1), stride=1, padding=0)
        self.k_proj = nn.Conv2d(features, features, kernel_size=(1, 1), stride=1, padding=0)
        self.v_proj = nn.Conv2d(features, features, kernel_size=(1, 1), stride=1, padding=0)
        self.linear = nn.Conv2d(features, features, kernel_size=(1, 1), stride=1, padding=0)
        

    def forward(self, x : Tensor) -> Tensor:
        '''
        x : [B, C, H, W]
        return : [B, C, H, W]
        '''
        h = x
        # print(self.features, h.shape)
        h = self.norm(h)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        B, C, H, W = x.shape
        # S = HW, F = C, s = S, S:For q, s:For k and v
        q = q.view(B, C, H*W) # [B, F, S]
        q = q.permute(0, 2, 1).contiguous() # [B, S, F]
        k = k.view(B, C, H*W) # [B, F, s]
        v = v.view(B, C, H*W) # [B, F, s]
        w = torch.einsum('BSF, BFs -> BSs', q, k) * (C ** (-0.5))
        w = F.softmax(w, dim=-1)
        w = w.permute(0, 2, 1).contiguous()
        h = torch.einsum('BFs, BsS -> BFS', v, w)
        h = h.view(B, C, H, W)
        h = self.linear(h)
        return x + h

class Encoder(nn.Module):
    def __init__(
        self,
        h_dims,
        in_res,
        attn_res,
        down_index,
        dropout,
        with_conv,
        act: nn.Module
    ) -> None:
        super().__init__()
        '''
        h_dims
        in_res
        attn_res
        down_index
        dropout_rate
        '''

        num_channels = len(h_dims) - 1
        f_res = in_res
        # Downsample
        self.Downsampling = nn.ModuleList()
        for index in range(0, num_channels):
            in_channels = h_dims[index]
            out_channels = h_dims[index+1]

            self.Downsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=act, dropout=dropout)
            )

            if f_res in attn_res:
                self.Downsampling.append(AttnBlock(features=out_channels))

            # Downsample if it is down index
            if down_index[index+1]==1:
                self.Downsampling.append(
                    DownBlock(in_features=out_channels, out_features=out_channels, act=act, with_conv=with_conv)
                )
                f_res /= 2

        self.f_res = f_res
    def forward(
        self,
        x : Tensor
    ):
        '''
        x : tensor
        return : tensor
        '''
        h = x
        for module in self.Downsampling:
            h = module(h)
        return h

class Decoder(nn.Module):
    def __init__(
        self,
        h_dims,
        in_res,
        attn_res,
        up_index,
        dropout,
        with_conv,
        act: nn.Module
    ) -> None:
        super().__init__()
        '''
        h_dims
        in_res
        attn_res
        up_index
        dropout_rate
        '''

        num_channels = len(h_dims) - 1
        f_res = in_res
        # Upsample
        self.Upsampling = nn.ModuleList()
        for index in range(0, num_channels):
            in_channels = h_dims[index]
            out_channels = h_dims[index+1]

            # Upsample if it is down index
            if up_index[index]==1:
                self.Upsampling.append(
                    UpBlock(in_features=in_channels, out_features=in_channels, act=act, with_conv=with_conv)
                )
                f_res *= 2

            if f_res in attn_res:
                self.Upsampling.append(AttnBlock(features=in_channels))

            self.Upsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=act, dropout=dropout)
            )
            

    def forward(
        self,
        x : Tensor
    ):
        '''
        x : tensor
        return : tensor
        '''
        h = x
        for module in self.Upsampling:
            h = module(h)
        return h

class AutoEncoderKL(pl.LightningModule):
    def __init__(
        self, 
        config
        ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.rec_factor, self.disc_factor = 1, 0.01
        self.config = config
        h_dims = config.get('h_dims')
        in_res = config.get('in_res')
        attn_res = config.get('attn_res')
        down_index = config.get('down_index')
        z_channels = config.get('z_channels')
        embed_dim = config.get('embed_dim')
        double_z = config.get('double_z')
        dropout = config.get('dropout')
        with_conv = config.get('with_conv')
        self.kld_factor = config.get('kld_factor')
        self.sampling_period = config.get('sampling_period')
        self.deterministic = config.get('deterministic')

        act = nn.ReLU()
        self.Encoder = Encoder(
            h_dims,
            in_res,
            attn_res,
            down_index,
            dropout,
            with_conv,
            act
        )
        # For encoder
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=h_dims[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(h_dims[-1], 
                                  2*z_channels if double_z else z_channels,
                                  kernel_size = 3,
                                  stride = 1,
                                  padding = 1
                                  )
        self.quant_conv = nn.Conv2d(2*z_channels, 2*embed_dim, kernel_size=(1, 1), stride=1, padding=0)

        in_res = self.Encoder.f_res
        reverse_h_dims = h_dims[::-1]
        up_index = down_index[::-1]
        # For decoder
        self.post_quant_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.conv_in = torch.nn.Conv2d(embed_dim,
                                       reverse_h_dims[0],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.Decoder = Decoder(
            reverse_h_dims,
            in_res,
            attn_res,
            up_index,
            dropout,
            with_conv,
            act
        )
        self.discriminator = NLayerDiscriminator()
        self.Distribution = DiagonalGaussianDistribution

    def forward(self, x: Tensor) -> Tensor:
        h = x
        posterior = self.encode(h)
        z = posterior.sample()
        dec = self.decode(z)
        return dec, posterior

    def encode(self, x: Tensor) -> Tensor:
        h = x
        h = self.Encoder(h)
        h = self.norm_out(h)
        h = self.conv_out(h)
        moments = self.quant_conv(h)
        posterior = self.Distribution(moments, deterministic=self.deterministic)
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        z = self.conv_in(z)
        dec = self.Decoder(z)
        return dec

    def disc_loss_func(self, x_rec, x, idx):
        logits_real = self.discriminator(x)
        logits_fake = self.discriminator(x_rec)

        bce_loss = nn.BCEWithLogitsLoss()

        loss_real = bce_loss(logits_real, torch.ones_like(logits_real))
        loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
        disc_loss = 0.5 * (loss_real + loss_fake)
        if idx==0: # VAE
            # Generator tries to minimize log(1 - D(G(z))), which is equivalent to maximizing log(D(G(z)))
            return bce_loss(logits_fake, torch.ones_like(logits_fake))
        if idx==1: # Discriminator
            # Discriminator tries to maximize log(D(x)) + log(1 - D(G(z)))
            return disc_loss

    def training_step(self, batch, batch_idx):
        x, _ = batch

        optimizer_ae, optimizer_disc = self.optimizers()

        self.toggle_optimizer(optimizer_ae)
        x_rec, posterior = self(x)
        x_disc = x.clone().detach()
        x_rec_disc = x_rec.clone().detach()
        # train encoder+decoder+logvar
        rec_loss = F.mse_loss(x_rec, x)
        kld_loss = posterior.kl().mean()
        disc_loss = self.disc_loss_func(x_rec, x, 0)
        ae_loss = self.rec_factor * rec_loss + self.kld_factor * kld_loss + self.disc_factor * disc_loss
        
        self.log("ae_loss", ae_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.manual_backward(ae_loss)
        optimizer_ae.step()
        optimizer_ae.zero_grad()
        self.untoggle_optimizer(optimizer_ae)
        # train the discriminator
        self.toggle_optimizer(optimizer_disc)
        disc_loss = self.disc_factor * self.disc_loss_func(x_rec_disc, x_disc, 1)
        self.manual_backward(disc_loss)
        optimizer_disc.step()
        optimizer_disc.zero_grad()
        self.untoggle_optimizer(optimizer_disc)
        self.log("disc_loss", disc_loss.detach() / self.disc_factor, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            x_rec, posterior = self(x)
            rec_loss = F.mse_loss(x_rec, x)
            kld_loss = posterior.kl().mean()
            disc_loss = self.disc_loss_func(x_rec, x, 0)
            loss = self.rec_factor * rec_loss + self.kld_factor * kld_loss + self.disc_factor * disc_loss
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        with torch.no_grad():
            x_rec, posterior = self(x)
            rec_loss = F.mse_loss(x_rec, x)
            kld_loss = posterior.kl().mean()
            disc_loss = self.disc_loss_func(x_rec, x, 0)
            loss = self.rec_factor * rec_loss + self.kld_factor * kld_loss + self.disc_factor * disc_loss
        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer_ae = \
        torch.optim.AdamW(
            list(self.Encoder.parameters())+
            list(self.Decoder.parameters())+
            list(self.quant_conv.parameters())+
            list(self.post_quant_conv.parameters()),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        optimizer_disc = \
        torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        optimizer = [optimizer_ae, optimizer_disc]
        if self.config['scheduler_gamma'] is not None:
            scheduler_ae = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer_ae, gamma = self.config['scheduler_gamma'])
            scheduler_disc = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer_disc, gamma = self.config['scheduler_gamma'])
            scheduler = [scheduler_ae, scheduler_disc]
            return optimizer, scheduler
        else:
            return optimizer, []

    def on_validation_epoch_end(self):
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()
    
    def sample(self, num_samples):
        
        z_shape = [num_samples] + [512, 56, 56]
        z = torch.randn(z_shape, device=self.device)
        # Decoder
        h = self.decode(z)
        return h

    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.device)
        test_label = test_label.to(self.device)

        recons = self(test_input)[0]
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)

        vutils.save_image(recons.data,
                          os.path.join(recons_dir,
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config, *args, **kwargs):
        model = cls(config, *args, **kwargs)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return model