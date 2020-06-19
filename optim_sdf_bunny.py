import math, os, sys

import numpy as np

import enoki as ek
import mitsuba
mitsuba.set_variant('gpu_autodiff_rgb')

from mitsuba.core import Thread, Float32
from mitsuba.core.xml import load_file, load_string
from mitsuba.python.util import traverse
from mitsuba.python.autodiff import render, render_torch, write_bitmap

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import skfmm
from scipy.ndimage import gaussian_filter

from utils import *
    
########################################
# Parameters
########################################

sdf_res = 64
sdf_res_squedule = [0, 50, 120, 210]
sdf_scale = 2.0
init_sphere_radius = 0.2

interpolation = "trilinear"
n_cams = 5
fov=27
img_res = 256

n_epochs = 400
restart_epoch = 0

lr_gamma = 0.995
lr = 0.008 * lr_gamma ** restart_epoch

increase_res = False
# sdf_res = 128
# lr *= 0.5

out_path = "output/optimisations/bunny/"

if len(sys.argv) > 1:
    out_path += sys.argv[1] + "/"
    print(f"Save into: {out_path}")

os.makedirs(out_path, exist_ok=True)

########################################
# SDF grid creation
########################################

transform = compose(translate([0.5, 0.5, 0.5]), scale(sdf_scale))
init_v = create_sdf(compose(transform, partial(sphere, radius=init_sphere_radius)), sdf_res)
write_binary_grid3d('data/sdf/init.vol', init_v)


########################################
# Rendering
########################################

def create_scene(sdf_file, interpolation, n_cams, fov, cam_res):
    scene = """
        <scene version='2.0.0'>

            {integrator}

            {sensors}
            
            <emitter type="point">
                <spectrum name="intensity" value="20"/>
                <point name="position" x="0" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="5" z="5"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="-5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="5" z="-5"/>
            </emitter>

            {shape}

            <shape type="obj">
                <string name="filename" value="data/meshes/ground.obj"/>
                <bsdf type="diffuse">
                    <!--<rgb name="reflectance" value="0.2, 0.25, 0.7"/>-->
                    <rgb name="reflectance" value="0.9, 0.9, 0.9"/>
                </bsdf>
            </shape>

        </scene>"""

    sdf = """
        <sdf id="SDF" type="sdf_explicit">
            <integer name="sphere_tracing_steps" value="150"/>
            <volume name="distance_field" type="gridvolume">
                <string name="interpolation_mode" value="$interpol"/>
                <transform name="to_world">
                    <translate x="-0.5" y="-0.3" z="-0.5"/>
                    <scale x="2" y="2" z="2"/>
                </transform>
                <boolean name="raw" value="true"/>
                <boolean name="use_grid_bbox" value="false"/>
                <string name="filename" value="$sdf_file"/>
            </volume>
            <bsdf type="diffuse">
                <!--<rgb name="reflectance" value="0.9, 0.9, 0.9"/>-->
                <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
            </bsdf>
        </sdf>
        """

    mesh = """
        <shape type="ply">
            <string name="filename" value="data/meshes/bunny.ply"/>
            <bsdf type="diffuse">
                <!--<rgb name="reflectance" value="0.9, 0.9, 0.9"/>-->
                <rgb name="reflectance" value="0.2, 0.25, 0.7"/>
            </bsdf>
        </shape>
        """

    sensor = """<sensor type="perspective">
            <float name="fov" value="$fov"/>
            <float name="near_clip" value="1"/>
            <float name="far_clip" value="1000"/>

            <transform name="to_world">
                <lookat target="0, 0.4, 0"
                        origin="0, 3, 3"
                        up    ="0, 1, 0"/>
                <rotate y="1" angle="{cam_angle}"/>
            </transform>

            <film type="hdrfilm">
                <rfilter type="box"/>
                <integer name="width" value="$res"/>
                <integer name="height" value="$res"/>
                <string name="pixel_format" value="rgb"/>
                <string name="component_format" value="float32"/>
            </film>

            <sampler type="independent">
                <integer name="sample_count" value="8"/>
            </sampler>
        </sensor>
        """

    integrator = """
        <integrator type='sdfpath'>
            <integer name="max_depth" value="5"/>
        </integrator>
        """ if sdf_file else """
        <integrator type='path'>
            <integer name="max_depth" value="5"/>
        </integrator>
        """
    sensors = " ".join([sensor.format(cam_angle=angle)
                        for i, angle in enumerate(np.linspace(0, 360, n_cams, endpoint=False))])
    scene = scene.format(sensors=sensors, shape=sdf if sdf_file else mesh, integrator=integrator)

    return load_string(scene, sdf_file=sdf_file, interpol=interpolation, n_cams=n_cams, fov=fov, res=cam_res)


scene_target = create_scene(None, interpolation, n_cams, fov, img_res)

images_ref = list(render_torch(scene_target, spp=64, sensor_index=i) for i in range(n_cams))
crop_size = scene_target.sensors()[0].film().crop_size()
for i in range(n_cams):
    write_bitmap(f'{out_path}{i}_target.png', images_ref[i], crop_size)

# Find differentiable scene parameters
sdf_file = "data/sdf/init.vol" if restart_epoch == 0 else f"{out_path}sdf_e{restart_epoch}.vol"
scene_init = create_scene(sdf_file, interpolation, n_cams, fov, img_res)
params = traverse(scene_init)

if restart_epoch > 0 and increase_res:
    params = traverse(scene_init)
    sdf = params['SDF.data'].numpy().reshape([sdf_res]*3)
    sdf = double_sdf_res(sdf) 
    sdf_res = sdf.shape[0]
    sdf_file = f'{out_path}sdf_e{restart_epoch}_x2.vol'
    write_binary_grid3d(sdf_file, sdf)
    scene_init = create_scene(sdf_file, interpolation, n_cams, fov, img_res)
    params = traverse(scene_init)

params.keep(['SDF.data'])
params_torch = params.torch()

opt = Adam(params_torch.values(), lr=lr)
lr_scheduler = ExponentialLR(opt, gamma=lr_gamma)

# loss = BoxPyramidLoss(images_ref, nn.MSELoss(reduction='mean'), 'mean')
loss = nn.MSELoss(reduction='mean')

with open(f"{out_path}log.txt", mode='a+') as f:
    f.write('epoch,lr,total_loss,' + ','.join([f'loss_img{i}' for i in range(n_cams)]) + "\n")

for epoch in range(restart_epoch, n_epochs):

    opt.zero_grad()

    loss_imgs = []
    for i in range(n_cams):
        print("render image", i)
        image = render_torch(scene_init, params=params, unbiased=True, spp=4, sensor_index=i, **params_torch)
        write_bitmap(f'{out_path}{i}_image_e{epoch:03d}.png', image, crop_size)

        # ob_val, _ = loss(image, i)
        ob_val = loss(image, images_ref[i])
        ob_val /= n_cams
        loss_imgs.append(ob_val.item())

        ob_val.backward()

    opt.step()

    try:
        sdf = params_torch['SDF.data'].data.cpu().numpy().reshape([sdf_res]*3)
        sdf = skfmm.distance(sdf, sdf_scale / sdf_res)

        params_torch['SDF.data'].data.copy_(torch.from_numpy(sdf.flatten()))

        if epoch % 10 == 9:
            write_binary_grid3d(f'{out_path}sdf_e{epoch}.vol', sdf)

    except RuntimeError as e:
        print(f'skfmm failed: mean={sdf.mean()}, min={sdf.min()}, max={sdf.max()}')
        print(e)

    print(f'epoch {epoch}: lr={lr_scheduler.get_lr()[0]}, total_loss={np.mean(loss_imgs)}, loss_imgs={loss_imgs}')
    with open(f"{out_path}log.txt", mode='a+') as f:
        f.write(','.join(list(map(str, [epoch, lr_scheduler.get_lr()[0], np.mean(loss_imgs), *loss_imgs, "\n"]))))

    lr_scheduler.step()