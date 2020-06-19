from utils import *

sdf_scale = 2

def create_cross_sdf(filename, smooth_r):
    print(f"Create SDF: {filename} ...")
    smooth_r += 1e-8
    transform = compose(translate([0.5, 0.5, 0.5]), scale(sdf_scale))
    cross = opRound(smooth_r, partial(box, dims=np.asarray([0.8, 0.11, 0.11])-smooth_r))
    sdf_vol = create_sdf(compose(transform,
                            opSmoothUnion(smooth_r,
                                compose(rot('y', 90).apply, cross),
                                opSmoothUnion(smooth_r,
                                    compose(rot('z', 90).apply, cross),
                                    cross
                                )
                            )
                        ), 128)

    write_binary_grid3d(f'data/sdf/{filename}', sdf_vol)


def create_sphere_cube_sdf(filename):
    print(f"Create SDF: {filename} ...")
    transform = compose(translate([0.5, 0.5, 0.5]), scale(sdf_scale))
    sdf_vol = create_sdf(compose(transform,
                            opUnion(
                                compose(partial(box, dims=[0.3]*3)),
                                compose(translate([0, 0.7, 0]), partial(sphere, radius=0.2)),
                            )
                        ), 128)

    write_binary_grid3d(f'data/sdf/{filename}', sdf_vol)


def create_sphere_sdf(filename):
    sphere_sdf = create_sdf(compose(
                    translate([0.5, 0.5, 0.5]), 
                    scale(sdf_scale), 
                    partial(sphere, radius=0.5)), 128)
    write_binary_grid3d(f'data/sdf/{filename}', sphere_sdf)


def create_cube_sdf(filename):
    sphere_sdf = create_sdf(compose(
                    translate([0.5, 0.5, 0.5]), 
                    scale(sdf_scale), 
                    partial(box, dims=[0.3]*3)), 128)
    write_binary_grid3d(f'data/sdf/{filename}', sphere_sdf)


# create_cross_sdf("cross.vol")
# create_sphere_cube_sdf("sc.vol")
create_sphere_sdf("sphere.vol")
create_cube_sdf("cube.vol")
