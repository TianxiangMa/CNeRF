import os
import torch
import trimesh
import numpy as np
import skvideo.io
import imageio
from scipy.interpolate import CubicSpline
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
from torchvision import utils
from torchvision import transforms
from options import BaseOptions
from model import Generator
from utils import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, add_textures,
    )
from pytorch3d.structures import Meshes
from dataset import color_segmap
from pdb import set_trace as st


def inference(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, semantics):
    g_ema.eval()
    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    images = torch.Tensor(0, 3, opt.size, opt.size)
    num_frames = 10
    trajectory = np.zeros((num_frames,3), dtype=np.float32)

    t1 = np.linspace(-1.5, 1.5, num_frames)
    t2 = 0.8 * np.ones(num_frames)

    fov = opt.camera.fov
    elev = opt.camera.elev * t2
    azim = opt.camera.azim * t1
    
    trajectory[:num_frames,0] = azim
    trajectory[:num_frames,1] = elev
    trajectory[:num_frames,2] = fov

    trajectory = torch.from_numpy(trajectory).to(device)
    
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
    generate_camera_params(opt.renderer_output_size, device, locations=trajectory[:,:2],
                           fov_ang=trajectory[:,2:], dist_radius=opt.camera.dist_radius)

    cameras = create_cameras(azim=np.rad2deg(trajectory[0,0].cpu().numpy()),
                             elev=np.rad2deg(trajectory[0,1].cpu().numpy()),
                             dist=1, device=device)
    renderer = create_mesh_renderer(cameras, image_size=512, specular_color=((0,0,0),),
                    ambient_color=((0.1,.1,.1),), diffuse_color=((0.75,.75,.75),),
                    device=device)
    
    # synthesis
    for i in range(opt.identities):
        print('Processing identity {}/{}...'.format(i+1, opt.identities))
        torch.cuda.empty_cache()
    
        styles = g_ema.style(torch.randn(1, opt.style_dim, device=device))
        styles = opt.truncation_ratio * styles + (1-opt.truncation_ratio) * mean_latent[0]
        styles_global = styles

        styles_new = styles.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        images, segs = [], []
        for j in range(0,num_frames):
            batch_size = 1
            for head in range(0, styles_new.size(0), batch_size):
                out = g_ema([styles_new[head:head+batch_size]],
                            sample_cam_extrinsics[j:j+1],
                            sample_focals[j:j+1],
                            sample_near[j:j+1],
                            sample_far[j:j+1],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent,
                            input_is_latent=True,
                            randomize_noise=False,
                            project_noise=opt.project_noise,
                            mesh_path=frontal_marching_cubes_mesh_filename if opt.project_noise else None,
                            styles_global=[styles_global.repeat(batch_size,1)],
                            semantics=semantics,
                            )

                images_, segs_ = out[0]
                
                del out
                torch.cuda.empty_cache()

                images_ = images_.clamp(-1,1) * 127.5 + 127.5
                images.append(images_.detach().cpu())
                segs_ = color_segmap(segs_)
                segs.append(segs_.detach().cpu())

        images = torch.cat(images,0)
        segs = torch.cat(segs,0)

        utils.save_image(images,
                        os.path.join(opt.results_dir, f"{str(i)}_img.png"),
                        nrow=num_frames,
                        normalize=True,)
        utils.save_image(segs,
                        os.path.join(opt.results_dir, f"{str(i)}_seg.png"),
                        nrow=num_frames,
                        normalize=True,)


        if not opt.no_surface_renderings:
            scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
            surface_sample_focals = sample_focals * scale
            surface_out = surface_g_ema([styles.unsqueeze(1).repeat(1, g_ema.n_latent, 1)],
                                        sample_cam_extrinsics[num_frames//2:num_frames//2+1],
                                        surface_sample_focals[num_frames//2:num_frames//2+1],
                                        sample_near[num_frames//2:num_frames//2+1],
                                        sample_far[num_frames//2:num_frames//2+1],
                                        truncation=opt.truncation_ratio,
                                        truncation_latent=surface_mean_latent,
                                        input_is_latent=True,
                                        randomize_noise=False,
                                        return_xyz=True,
                                        )
            xyz = surface_out[2].cpu()
            del surface_out
            torch.cuda.empty_cache()

            depth_mesh = xyz2mesh(xyz)
            mesh = Meshes(
                verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
                faces = [torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
                textures=None,
                verts_normals=[torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
            )
            mesh = add_textures(mesh)
            cameras = create_cameras(azim=np.rad2deg(trajectory[num_frames//2,0].cpu().numpy()),
                                    elev=np.rad2deg(trajectory[num_frames//2,1].cpu().numpy()),
                                    fov=2*trajectory[num_frames//2,2].cpu().numpy(),
                                    dist=1, device=device)
            renderer = create_mesh_renderer(cameras, image_size=256,
                                            light_location=((0.0,1.0,5.0),), specular_color=((0.2,0.2,0.2),),
                                            ambient_color=((0.1,0.1,0.1),), diffuse_color=((0.65,.65,.65),),
                                            device=device)

            mesh_image = 255 * renderer(mesh).cpu()
            mesh_image = mesh_image[...,:3]
            mesh_image = mesh_image.permute(0,3,1,2)

            utils.save_image(mesh_image,
                            os.path.join(opt.results_dir, f"{str(i)}_shape.png"),
                            nrow=1,
                            normalize=True,)
            
            del mesh, xyz, mesh_image
            torch.cuda.empty_cache()


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.style_dim = 256
    opt.model.freeze_renderer = False
    opt.rendering.depth = 3
    opt.rendering.width = 128
    opt.rendering.no_features_output = False
    opt.inference.size = opt.model.size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.rendering.perturb = 0
    opt.rendering.force_background = True
    opt.rendering.static_viewdirs = True
    opt.rendering.return_sdf = True
    opt.rendering.N_samples = 64

    os.makedirs(opt.inference.results_dir, exist_ok=True)
    checkpoint_path = opt.training.trained_ckpt
    checkpoint = torch.load(checkpoint_path)

    g_ema = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v
    g_ema.load_state_dict(model_dict)

    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model.copy()
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = 128
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        opt.inference.surf_extraction_output_size = 128
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)

        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v
        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None


    if opt.inference.truncation_ratio <= 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        mean_latent = None

    if opt.inference.truncation_ratio <= 1 and (not opt.inference.no_surface_renderings):
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    semantics = None
    if opt.inference.semantics != '':
        semantics = [int(s.strip()) for s in opt.inference.semantics.split(",")]

    inference(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, semantics)
