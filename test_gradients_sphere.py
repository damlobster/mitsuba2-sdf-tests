import numpy as np
import os
import mitsuba
import enoki as ek

mts_variant = 'rgb'
mitsuba.set_variant('gpu_autodiff_' + mts_variant)

from mitsuba.core import Transform4f, Bitmap
from mitsuba.core.xml import load_string
from mitsuba.python.util import traverse
from tests_utils import test_finite_difference

test_name = "test_gradients_sphere"

def make_scene(integrator, spp, res):
    # shape = """
    #     <sdf id="SDF" type="sdf_explicit">
    #         <integer name="sphere_tracing_steps" value="400"/>
    #         <volume name="distance_field" type="gridvolume">
    #             <string name="interpolation_mode" value="catmull-approx"/>
    #             <transform name="to_world">
    #                 <scale x="2" y="2" z="2"/>
    #                 <translate x="-1" y="-0.05" z="-1"/>
    #             </transform>
    #             <boolean name="raw" value="true"/>
    #             <boolean name="use_grid_bbox" value="false"/>
    #             <string name="filename" value="data/sdf/sphere.vol"/>
    #         </volume>
    #         <bsdf type="diffuse">
    #             <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
    #         </bsdf>
    #     </sdf>""" if fd_eps is None else f"""
    #     <shape type="obj">
    #         <string name="filename" value="data/meshes/sphere.obj"/>
    #         <transform name="to_world">
    #             <scale value="{0.685 - fd_eps}"/>
    #             <translate x="0.01" y="0.958" z="0.01"/>
    #         </transform>
    #         <bsdf type="diffuse">
    #             <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
    #         </bsdf>
    #     </shape>"""

    # shape = f"""
    #     <sdf id="SDF" type="sdf_sphere">
    #         <integer name="sphere_tracing_steps" value="400"/>
    #         <point name="center" value="0, 0.5, 0"/>
    #         <float name="radius" value="10"/>
    #         <bsdf type="diffuse">
    #             <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
    #         </bsdf>
    #     </sdf>"""

    shape = """
        <sdf id="SDF" type="sdf_explicit">
            <integer name="sphere_tracing_steps" value="400"/>
            <volume name="distance_field" type="gridvolume">
                <string name="interpolation_mode" value="catmull-approx"/>
                <transform name="to_world">
                    <scale x="2" y="2" z="2"/>
                    <translate x="-1" y="0.05" z="-1"/>
                </transform>
                <boolean name="raw" value="true"/>
                <boolean name="use_grid_bbox" value="false"/>
                <string name="filename" value="data/sdf/sphere.vol"/>
            </volume>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
            </bsdf>
        </sdf>"""

    return load_string(f"""
        <?xml version="1.0"?>
        <scene version="2.0.0">

            {integrator}
            {shape}

            <sensor type="perspective">
                <float name="fov" value="30"/>
                <float name="near_clip" value="1"/>
                <float name="far_clip" value="1000"/>

                <transform name="to_world">
                    <lookat target="0, 1, 0" 
                            origin="2.5, 3, 1.5"
                            up    ="0, 1, 0"/>
                </transform>

                <film type="hdrfilm">
                    <rfilter type="box"/>
                    <integer name="width" value="{res}"/>
                    <integer name="height" value="{res}"/>
                </film>

                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
            </sensor>

            <emitter type="point">
                <spectrum name="intensity" value="240"/>
                <point name="position" x="2" y="10" z="-2"/>
            </emitter>
            
            <!--<shape type="obj">
                <string name="filename" value="data/meshes/luminaire.obj"/>
                <emitter type="area">
                    <spectrum name="radiance" value="40"/>
                </emitter>
            </shape>-->

            <!--<shape type="obj">
                <string name="filename" value="data/meshes/unit_plane.obj"/>
                <transform name="to_world">
                    <rotate x="1" angle="90"/>
                    <scale value="2"/>
                    <translate x="-4" y="2" z="2"/>
                    <rotate y="1" angle="45"/>
                </transform>
                <bsdf type="conductor"></bsdf>
            </shape>-->

            <shape type="obj">
                <string name="filename" value="data/meshes/ground.obj"/>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="0.2, 0.2, 0.2"/>
                </bsdf>
            </shape>
        </scene>
    """)

image_res = 256

fd_spp = 64
fd_passes = 1
fd_integrator = """<integrator type="path">
                       <integer name="max_depth" value="2"/>
                   </integrator>"""

diff_spp = 2
diff_passes = 1
diff_integrator = """<integrator type="sdfpath">
                         <integer name="max_depth" value="2"/>
                     </integrator>"""

eps = 0.25 / 128

def _make_scene(integrator, spp):
    return make_scene(integrator, spp, image_res)

test_finite_difference(test_name, _make_scene,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, eps)