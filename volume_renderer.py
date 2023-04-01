import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from functools import partial
from pdb import set_trace as st


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, std_init=1, freq_init=False, is_first=False):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim))
        elif freq_init:
            self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25))
        else:
            self.weight = nn.Parameter(0.25 * nn.init.kaiming_normal_(torch.randn(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        self.bias = nn.Parameter(nn.init.uniform_(torch.empty(out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = self.std_init * F.linear(input, self.weight, bias=self.bias) + self.bias_init

        return out

# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3))
        else:
            self.weight = nn.Parameter(torch.empty(out_channel, in_channel).uniform_(-np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25))

        self.bias = nn.Parameter(nn.Parameter(nn.init.uniform_(torch.empty(out_channel), a=-np.sqrt(1/in_channel), b=np.sqrt(1/in_channel))))
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward(self, input, style):
        out = F.linear(input, self.weight, bias=self.bias)
        batch, features = out.size()[0], out.size()[-1]

        gamma = self.gamma(style).view(batch, 1, 1, 1, features)
        beta = self.beta(style).view(batch, 1, 1, 1, features)

        out = self.activation(gamma * out + beta)

        return out


# Siren Generator Model
class SirenGenerator(nn.Module):
    def __init__(self, D=3, W=128, style_dim=256, input_ch=3, input_ch_views=3, output_ch=4,
                 output_features=True, output_mask=True, init_net=False):
        super(SirenGenerator, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.style_dim = style_dim
        self.output_features = output_features
        self.output_mask = output_mask
        self.init_net = init_net

        self.shape_linears = nn.ModuleList(
            [FiLMSiren(3, W, style_dim=style_dim, is_first=True)] + \
            [FiLMSiren(W, W, style_dim=style_dim) for i in range(D-1)])

        self.texture_linears = nn.ModuleList(
            [FiLMSiren(W, W, style_dim=style_dim),
            FiLMSiren(W, W, style_dim=style_dim),]
        )
        
        self.mask_linears = LinearLayer(input_ch_views+W, 1, freq_init=True)
        
        self.feature_linear = LinearLayer(input_ch_views+W, W)

        self.rgb_linear = LinearLayer(W, 3, freq_init=True)
        self.sdf_linear = LinearLayer(W, 1, freq_init=True)


    def forward(self, x, styles):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        mlp_out = input_pts.contiguous()
        j = 0
        for i in range(len(self.shape_linears)):
            mlp_out = self.shape_linears[i](mlp_out, styles[:,j])
            j = j + 1
        mlp_out_mask = torch.cat([mlp_out, input_views], -1)
        mask = self.mask_linears(mlp_out_mask)
        for i in range(len(self.texture_linears)):
            mlp_out = self.texture_linears[i](mlp_out, styles[:,j])
            j = j + 1
        sdf = self.sdf_linear(mlp_out)
        if self.init_net:
            return sdf
        mlp_out_feature = torch.cat([mlp_out, input_views], -1)
        feature = self.feature_linear(mlp_out_feature)
        rgb = self.rgb_linear(feature)
        outputs = torch.cat([rgb, sdf], -1)
        
        if self.output_features:
            outputs = torch.cat([outputs, feature], -1)
        if self.output_mask:
            outputs = torch.cat([outputs, mask], -1)
        return outputs

# Full volume renderer
class VolumeFeatureRenderer(nn.Module):
    def __init__(self, opt, style_dim=256, out_im_res=64, mode='train'):
        super().__init__()
        self.test = mode != 'train'
        self.perturb = opt.perturb
        self.offset_sampling = not opt.no_offset_sampling
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.return_xyz = opt.return_xyz
        self.return_sdf = opt.return_sdf
        self.static_viewdirs = opt.static_viewdirs
        self.z_normalize = not opt.no_z_normalize
        self.out_im_res = out_im_res
        self.force_background = opt.force_background
        self.with_sdf = not opt.no_sdf
        if opt.no_features_output:
            self.output_features = False
        else:
            self.output_features = True

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        i, j = torch.meshgrid(torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res),
                              torch.linspace(0.5, self.out_im_res - 0.5, self.out_im_res))

        self.register_buffer('i', i.t().unsqueeze(0), persistent=False)
        self.register_buffer('j', j.t().unsqueeze(0), persistent=False)

        if self.offset_sampling:
            t_vals = torch.linspace(0., 1.-1/self.N_samples, steps=self.N_samples).view(1,1,1,-1)
        else:
            t_vals = torch.linspace(0., 1., steps=self.N_samples).view(1,1,1,-1)

        self.register_buffer('t_vals', t_vals, persistent=False)
        self.register_buffer('inf', torch.Tensor([1e10]), persistent=False)
        self.register_buffer('zero_idx', torch.LongTensor([0]), persistent=False)

        if self.test:
            self.perturb = False
            self.raw_noise_std = 0.

        self.channel_dim = -1
        self.samples_dim = 3
        self.input_ch = 3
        self.input_ch_views = 3
        self.feature_out_size = opt.width
        
        self.n_render = opt.n_render

        self.network = nn.ModuleList()
        for i in range(self.n_render):
            self.network.append(SirenGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch, output_ch=4, input_ch_views=self.input_ch_views, output_features=self.output_features))
        self.init_network = SirenGenerator(D=opt.depth, W=opt.width, style_dim=style_dim, input_ch=self.input_ch, output_ch=4, input_ch_views=self.input_ch_views, output_features=self.output_features, init_net=True)

        self.n_branch = opt.n_render
        self.shape_n = opt.depth
        self.texture_n = 2
        self.depth_total = self.shape_n + self.texture_n

    def get_rays(self, focal, c2w):
        dirs = torch.stack([(self.i - self.out_im_res * .5) / focal,
                            -(self.j - self.out_im_res * .5) / focal,
                            -torch.ones_like(self.i).expand(focal.shape[0], self.out_im_res, self.out_im_res)], -1)

        rays_d = torch.sum(dirs[..., None, :] * c2w[:,None,None,:3,:3], -1)
        rays_o = c2w[:,None,None,:3,-1].expand(rays_d.shape)
        if self.static_viewdirs:
            viewdirs = dirs
        else:
            viewdirs = rays_d

        return rays_o, rays_d, viewdirs

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(outputs=sdf, inputs=pts,
                                     grad_outputs=torch.ones_like(sdf),
                                     create_graph=True)[0]

        return eikonal_term

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def volume_integration(self, normalized_pts, viewdirs, styles, raw, z_vals, rays_d, pts, return_eikonal=False, semantics=None):
        dists = z_vals[...,1:] - z_vals[...,:-1]
        rays_d_norm = torch.norm(rays_d.unsqueeze(self.samples_dim), dim=self.channel_dim)
 
        dists = torch.cat([dists, self.inf.expand(rays_d_norm.shape)], self.channel_dim)
        dists = dists * rays_d_norm
        
        with torch.no_grad():
            input_dirs = viewdirs.unsqueeze(self.samples_dim).expand(normalized_pts.shape)
            net_inputs = torch.cat([normalized_pts, input_dirs], self.channel_dim)
            branch_latent = styles[:,:self.depth_total]
            init_sdf = self.init_network(net_inputs, styles=branch_latent).unsqueeze(0)

        if self.output_features:
            rgb, res_sdf, features, mask3d = torch.split(raw, [3, 1, self.feature_out_size, 1], dim=self.channel_dim)
        else:
            rgb, res_sdf, mask3d = torch.split(raw, [3, 1, 1], dim=self.channel_dim)

        sdf = init_sdf.squeeze(0) + torch.sum(res_sdf, 0)

        rgb = torch.sum(mask3d * rgb, 0, keepdim=False)

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)
            if return_eikonal:
                eikonal_term = self.get_eikonal_term(pts, sdf)
            else:
                eikonal_term = None
            sigma = 1 - torch.exp(-sigma * dists.unsqueeze(self.channel_dim))
        else:
            sigma = sdf
            eikonal_term = None
            sigma = 1 - torch.exp(-F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim))

        visibility = torch.cumprod(torch.cat([torch.ones_like(torch.index_select(sigma, self.samples_dim, self.zero_idx)), 1.-sigma + 1e-10], self.samples_dim), self.samples_dim)
        visibility = visibility[...,:-1,:]

        weights = sigma * visibility

        rgb = -1 + 2 * torch.sum(weights * torch.sigmoid(rgb), self.samples_dim)

        weights = weights.unsqueeze(0)
        
        self.samples_dim += 1
        mask2d = torch.sum(weights * mask3d, self.samples_dim)
        self.samples_dim -= 1

        weights = weights.squeeze(0)

        if self.output_features:
            # If you want to generate only some parts, you need to fuse the corresponding features.
            # face semantics = ['background','face','eye','brow','mouth','nose','ear','hair','neck+cloth']
            # e.g. features = mask3d[1] * features[1]
            if semantics is None:
                features = torch.sum(mask3d * features, 0, keepdim=False)
            elif isinstance(semantics, list):
                features_ = 0
                for s in semantics:
                    features_ = features_ + mask3d[s] * features[s]
                features = features_

            features = torch.sum(weights * features, self.samples_dim)
        else:
            features = None

        if self.return_sdf:
            sdf_out = sdf
        else:
            sdf_out = None

        if self.return_xyz:   
            xyz = torch.sum(weights * pts, self.samples_dim)
            mask = weights[...,-1,:]
        else:
            xyz = None
            mask = None

        return rgb, features, sdf_out, mask, xyz, eikonal_term, mask2d

    def run_network(self, inputs, viewdirs, styles=None, init=False):
        input_dirs = viewdirs.unsqueeze(self.samples_dim).expand(inputs.shape)
        net_inputs = torch.cat([inputs, input_dirs], self.channel_dim)

        n_outputs = []
        if init==False:
            for i in range(len(self.network)):
                branch_latent = styles[:, i*self.depth_total : (i+1)*self.depth_total]
                n_outputs.append(self.network[i](net_inputs, styles=branch_latent).unsqueeze(0))
        else:
            branch_latent = styles[:,:self.depth_total]
            n_outputs.append(self.init_network(net_inputs, styles=branch_latent).unsqueeze(0))

        outputs = torch.cat(n_outputs, dim=0)
        return outputs

    def render_rays(self, ray_batch, styles=None, return_eikonal=False, semantics=None):
        batch, h, w, _ = ray_batch.shape
        split_pattern = [3, 3, 2]
        if ray_batch.shape[-1] > 8:
            split_pattern += [3]
            rays_o, rays_d, bounds, viewdirs = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
        else:
            rays_o, rays_d, bounds = torch.split(ray_batch, split_pattern, dim=self.channel_dim)
            viewdirs = None

        near, far = torch.split(bounds, [1, 1], dim=self.channel_dim)
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        if self.perturb > 0.:
            if self.offset_sampling:
                upper = torch.cat([z_vals[...,1:], far], -1)
                lower = z_vals.detach()
                t_rand = torch.rand(batch, h, w).unsqueeze(self.channel_dim).to(z_vals.device)
            else:
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                t_rand = torch.rand(z_vals.shape).to(z_vals.device)

            z_vals = lower + (upper - lower) * t_rand
        
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)
        
        if return_eikonal:
            pts.requires_grad = True

        if self.z_normalize:
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts

        raw = self.run_network(normalized_pts, viewdirs, styles=styles, init=False)

        rgb, features, sdf, mask, xyz, eikonal_term, seg = self.volume_integration(normalized_pts.detach(), viewdirs.detach(), styles.detach(), raw, z_vals, rays_d, pts, return_eikonal=return_eikonal, semantics=semantics)

        return rgb, features, sdf, mask, xyz, eikonal_term, seg

    def render(self, focal, c2w, near, far, styles, c2w_staticcam=None, return_eikonal=False, semantics=None):
        rays_o, rays_d, viewdirs = self.get_rays(focal, c2w)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        rays = torch.cat([rays, viewdirs], -1)
        rays = rays.float()

        rgb, features, sdf, mask, xyz, eikonal_term, seg = self.render_rays(rays, styles=styles, return_eikonal=return_eikonal, semantics=semantics)

        return rgb, features, sdf, mask, xyz, eikonal_term, seg

    def mlp_init_pass(self, cam_poses, focal, near, far, styles=None):
        rays_o, rays_d, viewdirs = self.get_rays(focal, cam_poses)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

        near = near.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        far = far.unsqueeze(-1) * torch.ones_like(rays_d[...,:1])
        z_vals = near * (1.-self.t_vals) + far * (self.t_vals)

        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o.unsqueeze(self.samples_dim) + rays_d.unsqueeze(self.samples_dim) * z_vals.unsqueeze(self.channel_dim)
        if self.z_normalize:
            normalized_pts = pts * 2 / ((far - near).unsqueeze(self.samples_dim))
        else:
            normalized_pts = pts

        sdf = self.run_network(normalized_pts, viewdirs, styles=styles, init=True)
        
        sdf = sdf.squeeze(self.channel_dim).squeeze(0)
        target_values = pts.detach().norm(dim=-1) - ((far - near) / 4)

        return sdf, target_values

    def forward(self, cam_poses, focal, near, far, styles=None, return_eikonal=False, semantics=None):
        rgb, features, sdf, mask, xyz, eikonal_term, seg = self.render(focal, c2w=cam_poses, near=near, far=far, styles=styles, return_eikonal=return_eikonal, semantics=semantics)

        rgb = rgb.permute(0,3,1,2).contiguous()
        seg = seg.squeeze(-1).permute(1,0,2,3).contiguous()
        
        if self.output_features:
            features = features.permute(0,3,1,2).contiguous()

        if xyz != None:
            xyz = xyz.permute(0,3,1,2).contiguous()
            mask = mask.permute(0,3,1,2).contiguous()

        return rgb, features, sdf, mask, xyz, eikonal_term, seg
