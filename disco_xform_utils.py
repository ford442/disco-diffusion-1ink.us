import torch, torchvision
import py3d_tools as p3d
import midas_utils
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize,NormalizeImage,PrepareForNet
from PIL import Image
import numpy as np
import sys, math
import cv2
import torchvision.transforms as T
device=torch.device('cpu')
from numba import jit
try:
    from infer import InferenceHelper
except:
    print("disco_xform_utils.py failed to import InferenceHelper. Please ensure that AdaBins directory is in the path (i.e. via sys.path.append('./AdaBins') or other means).")
    sys.exit()

translate=(0.,0.,0.0)
near=0.2
far=16.0
fov_deg=114
padding_mode='border'
sampling_mode='bicubic'
midas_weight = 0.3
midas_model=DPTDepthModel(
            path='/content/midas/dpt_large-midas-2f21e586.pt',
            backbone="vitl16_384",
            non_negative=True,
)
net_w,net_h=384,384
resize_mode="minimal"
normalization=NormalizeImage(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
midas_transform=T.Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_LANCZOS4,
            ),
            normalization,
            PrepareForNet(),
])
MAX_ADABINS_AREA = 500000
MIN_ADABINS_AREA = 448*448

@torch.inference_mode()
#@jit(fastmath=True,forceobj=True,cache=True)
def transform_image_3d(img_filepath,imgsize):
    img_pil = Image.open(open(img_filepath, 'rb')).convert('RGB')
    w, h = img_pil.size
    image_tensor = torchvision.transforms.functional.to_tensor(img_pil).to(device)
    use_adabins = midas_weight < 1.0
    if use_adabins:
        infer_helper = InferenceHelper(dataset='nyu', device=device)
        image_pil_area = w*h
        if image_pil_area > MAX_ADABINS_AREA:
            scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC) # LANCZOS is supposed to be good for downsampling.
        elif image_pil_area < MIN_ADABINS_AREA:
            scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
            depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
        else:
            depth_input = img_pil
        try:
            _, adabins_depth = infer_helper.predict_pil(depth_input)
            if image_pil_area != MAX_ADABINS_AREA:
                adabins_depth = torchvision.transforms.functional.resize(torch.from_numpy(adabins_depth), image_tensor.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC).squeeze().to(device)
            else:
                adabins_depth = torch.from_numpy(adabins_depth).squeeze().to(device)
            adabins_depth_np = adabins_depth.cpu().numpy()
        except:
            pass

    #torch.cuda.empty_cache()
    # MiDaS
    img_midas = midas_utils.read_image(img_filepath)
    img_midas_input = midas_transform({"image": img_midas})["image"]
    midas_optimize = True
    # MiDaS depth estimation implementation
    #print("Running MiDaS depth estimation implementation...")
    sample = torch.from_numpy(img_midas_input).float().to(device).unsqueeze(0)
    sample = sample.to(memory_format=torch.channels_last)  
    #sample = sample.half()
    prediction_torch = midas_model.forward(sample)
    prediction_torch = torch.nn.functional.interpolate(
            prediction_torch.unsqueeze(1),
            size=img_midas.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    prediction_np = prediction_torch.clone().cpu().numpy()
    #print("Finished depth estimation.")
    #torch.cuda.empty_cache()
    # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
    prediction_np = np.subtract(50.0, prediction_np)
    prediction_np = prediction_np / 19.0
    adabins_weight = 1.0 - midas_weight
    depth_map = prediction_np*midas_weight + adabins_depth_np*adabins_weight
    depth_map = np.expand_dims(depth_map, axis=0)
    depth_tensor = torch.from_numpy(depth_map).squeeze().to(device)
    pixel_aspect = 1.0 # really.. the aspect of an individual pixel! (so usually 1.0)
    persp_cam_old = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    persp_cam_new = p3d.FoVPerspectiveCameras(near, far, pixel_aspect, fov=fov_deg, degrees=True, device=device)
    y,x = torch.meshgrid(torch.linspace(-1.,1.,h,dtype=torch.float32,device=device),torch.linspace(-1.,1.,w,dtype=torch.float32,device=device))
    z = torch.as_tensor(depth_tensor, dtype=torch.float32, device=device)
    xyz_old_world = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    xyz_old_cam_xy = persp_cam_old.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    xyz_new_cam_xy = persp_cam_new.get_full_projection_transform().transform_points(xyz_old_world)[:,0:2]
    offset_xy = xyz_new_cam_xy - xyz_old_cam_xy
    identity_2d_batch = torch.tensor([[1.,0.,0.],[0.,1.,0.]], device=device).unsqueeze(0)
    coords_2d = torch.nn.functional.affine_grid(identity_2d_batch, [1,1,h,w], align_corners=False)
    offset_coords_2d = coords_2d - torch.reshape(offset_xy, (h,w,2)).unsqueeze(0)

    new_image = torch.nn.functional.grid_sample(image_tensor.add(1/512 - 0.0001).unsqueeze(0), offset_coords_2d, mode=sampling_mode, padding_mode=padding_mode, align_corners=False)

    img_pil = torchvision.transforms.ToPILImage()(new_image.squeeze().clamp(0,1.))

    #torch.cuda.empty_cache()

    return img_pil

def get_spherical_projection(H, W, center, magnitude,device):  
    xx, yy = torch.linspace(-1, 1, W,dtype=torch.float32,device=device), torch.linspace(-1, 1, H,dtype=torch.float32,device=device)  
    gridy, gridx  = torch.meshgrid(yy, xx)
    grid = torch.stack([gridx, gridy], dim=-1)  
    d = center - grid
    d_sum = torch.sqrt((d**2).sum(axis=-1))
    grid += d * d_sum.unsqueeze(-1) * magnitude 
    return grid.unsqueeze(0)
