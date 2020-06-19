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

#sdf_res = 64
sdf_res = 128
#sdf_res_squedule = [0, 50, 120, 210]
sdf_scale = 2.0
init_sphere_radius = 0.4
interpolation = "trilinear"
color_texture_res = 4

# cams_origins = ["0, 4, 0.001", "-2, 4, 3", "2, 4, 3", "-2, 4, -3", "2, 4, -3"]
# fov=[35, 23, 23, 23, 23]
cams_origins = ["0, 4, 0.001", "-2, 4, 3", "2, 4, 3", "-2, 4, -3", "2, 4, -3", "0, 4, 3", "0, 4, -3"]
fov=[35, 23, 23, 23, 23, 30, 30]
img_res = 256

restart_epoch = 989
lr_gamma=1.0 #0.997 #TODO
lr=0.002 #0.003 * lr_gamma**restart_epoch # TODO
n_epochs = 1000
increase_res = False

out_path = "output/optimisations/bunny_dragon/"

if len(sys.argv) > 1:
    out_path += sys.argv[1] + "/"
    print(f"Save into: {out_path}")

os.makedirs(out_path, exist_ok=True)

########################################
# SDF grid creation
########################################

if restart_epoch == 0:
    transform = compose(translate([0.5, 0.3, 0.5]), scale(sdf_scale))
    #init_v = create_sdf(compose(transform, partial(box, dims=[0.08, 0.08, 0.4])), sdf_res)
    init_v = create_sdf(compose(transform,
                                opUnion(
                                    compose(translate([0, 0, 0.5]), partial(sphere, radius=0.15)),
                                    compose(translate([-0.3, 0, -0.4]), partial(sphere, radius=0.15))
                                )), sdf_res)
    write_binary_grid3d('data/sdf/init.vol', init_v)

########################################
# Rendering
########################################

def create_scene(sdf_file, interpolation, n_cams, fov, cam_res):
    scene = """
        <scene version='2.0.0'>

            {integrator}

            {sensors}
            
            <!--<emitter type="envmap">
                <string name="filename" value="data/textures/studio.exr"/>
                <float name="scale" value="0.8"/>
            </emitter>-->

            <!--<emitter type="point">
                <spectrum name="intensity" value="20"/>
                <point name="position" x="0" y="5" z="0"/>
            </emitter>-->
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="2.5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="-2.5" y="5" z="0"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="5" z="2.5"/>
            </emitter>
            <emitter type="point">
                <spectrum name="intensity" value="30"/>
                <point name="position" x="0" y="5" z="-2.5"/>
            </emitter>

            {shape}

            <shape type="obj">
                <string name="filename" value="data/meshes/ground.obj"/>
                <bsdf type="diffuse">
                    <texture name="reflectance" type="checkerboard">
                        <rgb name="color0" value="0.7, 0.7, 0.7"/>
                        <rgb name="color1" value="0.8, 0.8, 0.8"/>
                        <transform name="to_uv">
                            <scale x="10" y="10"/>
                        </transform>
                    </texture>
                </bsdf>
            </shape>

        </scene>"""


    if restart_epoch == 0:
        color = f"data/textures/{color_texture_res}x{color_texture_res}_gray.png" 
    else:
        color = f"{out_path}color_e{restart_epoch:03d}.exr"

    sdf = f"""
        <sdf id="SDF" type="sdf_explicit">
            <integer name="sphere_tracing_steps" value="150"/>
            <volume name="distance_field" type="gridvolume">
                <string name="interpolation_mode" value="$interpol"/>
                <transform name="to_world">
                    <translate x="-0.5" y="-0.15" z="-0.5"/>
                    <scale x="2" y="2" z="2"/>
                </transform>
                <boolean name="raw" value="true"/>
                <boolean name="use_grid_bbox" value="false"/>
                <string name="filename" value="$sdf_file"/>
            </volume>
            <bsdf type="diffuse">
                <texture type="bitmap" name="reflectance">
                    <string name="filename" value="{color}"/> <!--uv_lr.png-->
                    <transform name="to_uv">
                        <translate x="0.25"/>
                    </transform>
                </texture>
            </bsdf>
        </sdf>
        """

    mesh = """
        <shape type="ply">
            <string name="filename" value="data/meshes/bunny.ply"/>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.5, 0.05, 0.05"/>
            </bsdf>
            <transform name="to_world">
                <scale value="0.95"/>
                <rotate y="1" angle="180"/>
                <translate x="-0.1" z="0.4"/>
            </transform>
        </shape>
        <shape type="obj">
            <string name="filename" value="data/meshes/dragon.obj"/>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.05, 0.45, 0.05"/>
            </bsdf>
            <transform name="to_world">
                <translate x="0.07" z="-0.4"/>
            </transform>
        </shape>
        """

    sensor = """<sensor type="perspective">
            <float name="fov" value="{fov}"/>
            <float name="near_clip" value="0.1"/>
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
                <integer name="sample_count" value="16"/>
            </sampler>
        </sensor>
        """

    integrator = """
        <integrator type='sdfpath'>
            <integer name="max_depth" value="4"/>
        </integrator>
        """ if sdf_file else """
        <integrator type='path'>
            <integer name="max_depth" value="4"/>
        </integrator>
        """

    sensors = " ".join([sensor.format(cam_origin=o, fov=f)
                        for o, f in zip(cams_origins, fov)])
    scene = scene.format(sensors=sensors, shape=sdf if sdf_file else mesh, integrator=integrator, color=color)

    print(f"Using SDF: {sdf_file}, color={color}")
    return load_string(scene, sdf_file=sdf_file, interpol=interpolation, n_cams=n_cams, res=cam_res)

n_cams = len(cams_origins)
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

params = traverse(scene_init)
params.keep(['SDF.data', 'SDF.bsdf.reflectance.data'])

params_torch = params.torch()

opt = Adam(params_torch.values(), lr=lr)
lr_scheduler = ExponentialLR(opt, gamma=lr_gamma)

loss = nn.MSELoss(reduction="mean") #BoxPyramidLoss(images_ref, nn.MSELoss(reduction="mean"), 'mean')
vtk = VtkWriter(out_path + "vtk/sdf_grad", sdf_res, sdf_scale)

for epoch in range(restart_epoch+1, n_epochs):

    opt.zero_grad()

    loss_img = 0
    #pyramid_loss = 0
    for i in range(n_cams):
        print("render image", i)
        image = render_torch(scene_init, params=params, unbiased=True, spp=4, sensor_index=i, **params_torch)
        write_bitmap(f'{out_path}{i}_image_e{epoch:03d}.png', image, crop_size)

        # ob_val, pl = loss(image, i)
        # ob_val /= n_cams
        ob_val = loss(image, images_ref[i])
        ob_val /= n_cams
        loss_img += ob_val.item()
        # pyramid_loss += pl.data
        ob_val.backward()

    grad = params_torch['SDF.data'].grad.cpu().numpy().reshape([sdf_res]*3)
    opt.step()
    lr_scheduler.step()

    params_torch['SDF.bsdf.reflectance.data'].data.clamp_(0.0, 1.0)
    try:
        sdf = params_torch['SDF.data'].data.cpu().numpy().reshape([sdf_res]*3)
        sdf = skfmm.distance(sdf, sdf_scale / sdf_res)

        vtk.record_epoch(epoch, sdf, grad)

        params_torch['SDF.data'].data.copy_(torch.from_numpy(sdf.flatten()))

        if epoch % 10 == 9:
            write_binary_grid3d(f'{out_path}sdf_e{epoch}.vol', sdf)
            write_bitmap(f'{out_path}color_e{epoch:03d}.exr', params['SDF.bsdf.reflectance.data'], [color_texture_res]*2)

    except RuntimeError as e:
        print(f'skfmm failed: mean={sdf.mean()}, min={sdf.min()}, max={sdf.max()}')
        print(e)

    with open(f"{out_path}log.txt", mode='a+') as f:
        # f.write(','.join(list(map(str, [epoch, lr_scheduler.get_lr()[0], loss_img, *(pyramid_loss.cpu().numpy()), '\n']))))
        f.write(','.join(list(map(str, [epoch, lr_scheduler.get_lr()[0], loss_img, "\n"]))))

    print(f'epoch {epoch}: loss_img={loss_img}')



