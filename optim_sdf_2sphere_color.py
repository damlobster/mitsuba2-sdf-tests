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
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

import skfmm
from scipy.ndimage import gaussian_filter

from utils import *


########################################
# Parameters
########################################

sdf_res = 64
sdf_res_squedule = [0, 50, 120, 210]
sdf_scale = 2.0
init_sphere_radius = 0.35

interpolation = "trilinear"
# cams_origins = ["0, 4.65, 0.001", "0, 3, 3", "0, 3, -3"]
cams_origins = ["0, 4.65, 0.001", "0, 0.5, 3", "0, 0.5, -3"]
fov=27
img_res = 256
spp=4
spp_ref=64

n_epochs = 400
lr=0.01
lr_gamma=0.995
lambda_img = 1.0

restart_epoch = 0 #229
#lr = 0.01 * 0.99**100
increase_res = False

out_path = "output/optimisations/optim_2sphere_color/"
if len(sys.argv) > 1:
    out_path += sys.argv[1] + "/"
    print(f"Save into: {out_path}")

os.makedirs(out_path, exist_ok=True)

########################################
# SDF grid creation
########################################

transform = compose(translate([0.5, 0.5, 0.5]), scale(sdf_scale))
# init_v = create_sdf(compose(transform, partial(sphere, radius=init_sphere_radius)), sdf_res)
init_v = create_sdf(compose(transform, partial(box, dims=[0.4, 0.08, 0.08])), sdf_res)
# init_v = create_sdf(compose(transform, intersect(
#                                 compose(translate([-0.4, 0, 0]), partial(sphere, radius=init_sphere_radius)),
#                                 compose(translate([0.4, 0, 0]), partial(sphere, radius=init_sphere_radius)))
#                             ), sdf_res)

write_binary_grid3d('data/sdf/init.vol', init_v)


########################################
# Rendering
########################################

def create_scene(sdf_file, interpolation, cams_origins, fov, cam_res):
    scene = """
        <scene version='2.0.0'>

            {integrator}

            {sensors}
            
            <emitter type="point">
                <spectrum name="intensity" value="25"/>
                <point name="position" x="0" y="5" z="0"/>
            </emitter>
            <!--<emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="1" z="5"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="-5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="1" z="-5"/>
            </emitter>-->

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
                <texture type="bitmap" name="reflectance">
                    <string name="filename" value="data/textures/4x4_gray.png"/> <!--uv_lr.png-->
                    <transform name="to_uv">
                        <translate x="0.25"/>
                    </transform>
                </texture>
            </bsdf>
        </sdf>
        """

    mesh = """
        <shape type="obj">
            <string name="filename" value="data/meshes/sphere_small.obj"/>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.8, 0.2, 0.2"/>
            </bsdf>
            <transform name="to_world">
                <scale x="0.4" y="0.4" z="0.4"/>
                <translate x="-0.4" y="0.5" z="0"/>
            </transform>
        </shape>
        <shape type="obj">
            <string name="filename" value="data/meshes/sphere_small.obj"/>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.2, 0.2, 0.8"/>
            </bsdf>
            <transform name="to_world">
                <scale x="0.4" y="0.4" z="0.4"/>
                <translate x="0.4" y="0.5" z="0"/>
            </transform>
        </shape>
        """

    sensor = """<sensor type="perspective">
            <float name="fov" value="$fov"/>
            <float name="near_clip" value="1"/>
            <float name="far_clip" value="1000"/>

            <transform name="to_world">
                <lookat target="0, 0.4, 0"
                        origin="{cam_origin}"
                        up    ="0, 1, 0"/>
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
            <integer name="max_depth" value="3"/>
        </integrator>
        """ if sdf_file else """
        <integrator type='path'>
            <integer name="max_depth" value="3"/>
        </integrator>
        """
    sensors = " ".join([sensor.format(cam_origin=o)
                        for o in cams_origins])
    scene = scene.format(sensors=sensors, shape=sdf if sdf_file else mesh, integrator=integrator)

    return load_string(scene, sdf_file=sdf_file, interpol=interpolation, fov=fov, res=cam_res)


scene_target = create_scene(None, interpolation, cams_origins, fov, img_res)

images_ref = list(render_torch(scene_target, spp=spp_ref, sensor_index=i) for i in range(len(cams_origins)))
crop_size = scene_target.sensors()[0].film().crop_size()
for i in range(len(cams_origins)):
    write_bitmap(f'{out_path}{i}_target.png', images_ref[i], crop_size)

# Find differentiable scene parameters
sdf_file = "data/sdf/init.vol" if restart_epoch == 0 else f"{out_path}sdf_e{restart_epoch}.vol"
scene_init = create_scene(sdf_file, interpolation, cams_origins, fov, img_res)
params = traverse(scene_init)
#print(params)
params.keep(['SDF.data', 'SDF.bsdf.reflectance.data'])

if restart_epoch > 0 and increase_res:
    sdf = params['SDF.data'].numpy().reshape([sdf_res]*3)
    sdf = double_sdf_res(sdf) 
    params['SDF.data'] = sdf.flatten()
    params.update()

params_torch = params.torch()

opt = Adam(params_torch.values(), lr=lr)
lr_scheduler = ExponentialLR(opt, gamma=lr_gamma)

T_max = 0
#lr_scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=0.001)

objective = BoxPyramidLoss(images_ref, nn.MSELoss(reduction="mean"))

for epoch in range(restart_epoch, n_epochs):

    opt.zero_grad()

    loss_img = 0
    for i in range(len(cams_origins)):
        image = render_torch(scene_init, params=params, unbiased=True, spp=spp, sensor_index=i, **params_torch)
        write_bitmap(f'{out_path}{i}_image_e{epoch:03d}.png', image, crop_size)

        #ob_val = lambda_img * objective(image, images_ref[i]) / len(cams_origins)
        ob_val, pyr_ob = objective(image, i)
        print(f"rendered image {i}: loss={ob_val.cpu().item()}, pyramid_losses={list(pyr_ob.data.cpu().numpy())}")

        ob_val /= len(cams_origins)
        loss_img += ob_val.item()
        ob_val.backward()


    opt.step()
    if lr_scheduler:
        print("lr = ", lr_scheduler.get_lr()[0])
        lr_scheduler.step()
        if epoch == T_max and T_max > 0:
            lr_scheduler = CosineAnnealingLR(opt, T_max=T_max, eta_min=0.001)

    print(ek.hsum(params['SDF.bsdf.reflectance.data']))

    try:
        sdf = params_torch['SDF.data'].data.cpu().numpy().reshape([sdf_res]*3)
        sdf = skfmm.distance(sdf, sdf_scale / sdf_res)

        if epoch in sdf_res_squedule:
            pass
            # print(sdf.shape, sdf.flatten().shape)
            # sdf = double_sdf_res(sdf) 
            # print(sdf.shape, sdf.flatten().shape)
            # params_torch['SDF.data'].data = torch.from_numpy(sdf.flatten()).cuda()
            # # params['SDF.data'] = sdf.flatten()
            # params.update()
        else:
            params_torch['SDF.data'].data.copy_(torch.from_numpy(sdf.flatten()))

        if epoch % 10 == 0:
            write_binary_grid3d(f'{out_path}sdf_e{epoch}.vol', sdf)
            write_bitmap(f'{out_path}color_e{epoch:03d}.png', params['SDF.bsdf.reflectance.data'], [4,4])

    except RuntimeError as e:
        print(f'skfmm failed: mean={sdf.mean()}, min={sdf.min()}, max={sdf.max()}')
        print(e)

    print(f'epoch {epoch}: loss_img={loss_img}')

