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

test_name = "test_gradients_direct_visibility"

def make_scene(integrator, spp, res, sdf_vol):
    return load_string(f"""
        <?xml version="1.0"?>
        <scene version="2.0.0">

            {integrator}

            <sensor type="perspective">
                <float name="fov" value="30"/>
                <float name="near_clip" value="1"/>
                <float name="far_clip" value="1000"/>

                <transform name="to_world">
                    <lookat target="0, 1.1, 0" 
                            origin="2.5, 3, 1.5"
                            up    ="0, 1, 0"/>
                </transform>

                <film type="hdrfilm">
                    <rfilter type="box"/>
                    <integer name="width" value="{res}"/>
                    <integer name="height" value="{res}"/>
                    <string name="pixel_format" value="rgb"/>
                    <string name="component_format" value="float32"/>
                </film>

                <sampler type="independent">
                    <integer name="sample_count" value="{spp}"/>
                </sampler>
            </sensor>

            <sdf id="SDF" type="sdf_explicit">
                <integer name="sphere_tracing_steps" value="400"/>
                <volume name="distance_field" type="gridvolume">
                    <string name="interpolation_mode" value="trilinear"/>
                    <transform name="to_world">
                        <scale x="2" y="2" z="2"/>
                        <translate x="-1" y="0.05" z="-1"/>
                    </transform>
                    <boolean name="raw" value="true"/>
                    <boolean name="use_grid_bbox" value="true"/>
                    <string name="filename" value="data/sdf/{sdf_vol}"/>
                </volume>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="0.8, 0.8, 0.8"/>
                </bsdf>
            </sdf>

            <emitter type="point">
                <spectrum name="intensity" value="2500"/>
                <point name="position" x="10" y="40" z="20"/>
            </emitter>

            <shape type="obj">
                <string name="filename" value="data/meshes/ground.obj"/>
                <bsdf type="diffuse">
                    <rgb name="reflectance" value="0.2, 0.2, 0.2"/>
                </bsdf>
                <!--<bsdf type="roughconductor">
                    <float name="alpha" value="0.1"/>
                </bsdf>-->
            </shape>
        </scene>
    """)

image_res = 200

def make_scene_(integrator, spp):
    return make_scene(integrator, spp, image_res, "cube.vol")

fd_spp = 100
fd_passes = 10
fd_integrator = """<integrator type="path">
                       <integer name="max_depth" value="2"/>
                   </integrator>"""

diff_spp = 4
diff_passes = 10
diff_integrator = """<integrator type="sdfpath">
                         <integer name="max_depth" value="2"/>
                     </integrator>"""

eps = 0.25 / 128

test_finite_difference(test_name, make_scene_,
    diff_integrator, diff_spp, diff_passes,
    fd_integrator, fd_spp, fd_passes, eps)