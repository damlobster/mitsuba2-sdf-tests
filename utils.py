import os
from functools import partial, reduce
import math
from math import sqrt, sin, cos, ceil
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import skfmm

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from mitsuba.core import Float, UInt32

import pyevtk
from pyevtk.hl import imageToVTK
from pyevtk.vtk import VtkGroup

######################################################
## SDF volumes generation
######################################################

def write_binary_grid3d(filename, vals):
    print(f"Save to: {filename}, shape = {vals.shape}, min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")
    with open(filename, 'wb') as f:
        f.write(b'V')
        f.write(b'O')
        f.write(b'L')
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(vals.shape[0]).tobytes())  # size
        f.write(np.int32(vals.shape[1]).tobytes())
        f.write(np.int32(vals.shape[2]).tobytes())
        if vals.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(vals.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(vals.ravel().astype(np.float32).tobytes())

def create_sdf(func, res):
    v = np.zeros((res, res, res))
    for z in range(res):
        for y in range(res):
            for x in range(res):
                v[z, y, x] = func(np.asarray([x/res, y/res, z/res]))
    return v

def double_sdf_res(sdf):
    interp = RegularGridInterpolator(tuple([np.linspace(0, 1, num=n) for n in sdf.shape]), sdf, bounds_error=False)
    nx, ny, nz = np.asarray(sdf.shape)*2
    x, y, z = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), np.linspace(0, 1, nz))
    xyz = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    return interp(xyz.reshape(nx, ny, nz, 3)).transpose((1, 0, 2))

# SDF transformations

def compose(*funcs):
    def c(f, g):
        return lambda x: g(f(x))
    return reduce(c, funcs, lambda x: x)

def rot(axis, angle):
    return R.from_euler(axis, angle, degrees=True)

def translate(vec):
    return lambda p: p-np.asarray(vec)

def scale(s):
    return lambda p: p*s

def opUnion(sdf1, sdf2):
    return lambda p: min(sdf1(p), sdf2(p))

def opSubtraction(sdf1, sdf2):
    return lambda p: max(-sdf1(p), sdf2(p))

def opIntersection(sdf1, sdf2):
    return lambda p: max(sdf1(p), sdf2(p))

def opSmoothUnion(k, sdf1, sdf2):
    def op(p):
        d1, d2 = sdf1(p), sdf2(p)
        h = max(k-abs(d1-d2),0.0)
        return min(d1, d2) - h*h*0.25/k
    return op

def opSmoothSubtraction(k, sdf1, sdf2):
    def op(p):
        d1, d2 = sdf1(p), sdf2(p)
        h = max(k-abs(-d1-d2),0.0)
        return max(-d1, d2) + h*h*0.25/k
    return op

def opSmoothIntersection(k, sdf1, sdf2):
    def op(p):
        d1, d2 = sdf1(p), sdf2(p)
        h = max(k-abs(d1-d2),0.0)
        return max(d1, d2) + h*h*0.25/k
    return op

def opRound(radius, sdf):
    return lambda p: sdf(p) - radius

def opContract(sdf, radius):
    return lambda p: sdf(p) + radius


def opOnion(sdf, thickness):
    return lambda p: abs(sdf(p))-thickness

# Analytical SDF functions

def sphere(p, radius):
    d = np.linalg.norm(p) - radius
    return d

def box(p, dims=np.array([0.25, 0.25, 0.25])):
    q = np.abs(p) - dims
    return np.linalg.norm(np.clip(q, 0.0, np.inf)) + min(max(q[0],max(q[1],q[2])),0.0)

def torus(p, r1=0.25, r2=0.1):
    x, y, z = p
    q = sqrt(x**2 + y**2) - r1
    return sqrt(q**2 + z**2) - r2

def cylinder(p, h, r):
    d = np.abs(np.asarray([norm([p[0], p[2]]), p[1]])) - np.asarray([r, h])
    return min(max(d[0],d[1]),0.0) + norm(np.clip(d, 0.0, np.inf))


######################################################
## pytorch utils
######################################################


class Smoothing(nn.Module):
    def __init__(self, grid_size, kernel_size, clip, boundary='constant'):
        super(Smoothing, self).__init__()

        kernel_size += 1 - kernel_size % 2
        self.grid_size = grid_size
        sigma = kernel_size / 4
        padding = kernel_size // 2
        kernel_size = [kernel_size]*3
        self.clip = clip

        kernel = 1
        meshgrids = torch.meshgrid([
                    torch.arange(size, dtype=torch.float32)
                    for size in kernel_size
                ])

        for size, std, mgrid in zip(kernel_size, [sigma]*3, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)

        kernel = torch.nn.Parameter(kernel.view(1, 1, *kernel.size()), requires_grad=False)
        #self.register_buffer('weight', kernel)
        #self.boundary = boundary
        self.conv = nn.Conv3d(1, 1, kernel.size(), padding=padding, padding_mode='relfect', bias=False)
        self.conv.weight = kernel

    def forward(self, input):
        input = input.view(1, 1, *self.grid_size)
        # output = F.pad(input, pad=self.padding, mode=self.boundary, value=10)
        # output = F.conv3d(input, weight=self.weight, groups=1)
        with torch.no_grad():
            output = self.conv(input)
            if self.clip:
                output = torch.max(torch.min(input, output + self.clip), output - self.clip)
            output = output.view(-1).clone() # to make view memory contiguous
        return output


def eikonal_loss(grid, world_size, grid_res):
    # this code come from SDFDiff repository
    
    n = grid_res - 1
    voxel_size = world_size / n

    grid = grid.view([grid_res]*3)

    # x-axis normal vectors
    X_1 = torch.cat((grid[1:,:,:], (3 * grid[n,:,:] - 3 * grid[n-1,:,:] + grid[n-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid[1,:,:] + 3 * grid[0,:,:] + grid[2,:,:]).unsqueeze_(0), grid[:n,:,:]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    # y-axis normal vectors
    Y_1 = torch.cat((grid[:,1:,:], (3 * grid[:,n,:] - 3 * grid[:,n-1,:] + grid[:,n-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid[:,1,:] + 3 * grid[:,0,:] + grid[:,2,:]).unsqueeze_(1), grid[:,:n,:]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)

    # z-axis normal vectors
    Z_1 = torch.cat((grid[:,:,1:], (3 * grid[:,:,n] - 3 * grid[:,:,n-1] + grid[:,:,n-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid[:,:,1] + 3 * grid[:,:,0] + grid[:,:,2]).unsqueeze_(2), grid[:,:,:n]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)

    grad_sq = torch.pow(grid_normal_x[1:grid_res-1, 1:grid_res-1, 1:grid_res-1], 2)\
            + torch.pow(grid_normal_y[1:grid_res-1, 1:grid_res-1, 1:grid_res-1], 2)\
            + torch.pow(grid_normal_z[1:grid_res-1, 1:grid_res-1, 1:grid_res-1], 2)

    loss = torch.mean(torch.abs(grad_sq - 1))
    return loss


class BoxPyramidLoss():
    def __init__(self, ref_images, criterion, reduction='mean', reweight=True):
        self.ref_images = ref_images
        self.criterion = criterion
        self._reduction = torch.mean if reduction == 'mean' else torch.sum
        self._reweight = reweight

        with torch.no_grad():
            self.ref_pyramids = list(self._build_pyramid(ref_image, 0) for ref_image in ref_images)

    def _build_pyramid(self, image, from_level):
        pyramid = []

        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        box_size = 2<<from_level
        if from_level > 0:
            assert box_size <= image.size()[2], f"box_size={box_size} > image.size={image.size()[2]}"
            pyramid.append(torch.nn.functional.avg_pool2d(image, box_size, ceil_mode=True))
        else:
            pyramid.append(image)
        
        while pyramid[-1].size()[1] > 1 and pyramid[-1].size()[2] > 1:
            box_size *= 2
            pyramid.append(torch.nn.functional.avg_pool2d(image, box_size, ceil_mode=True))
        
        return pyramid

    def __call__(self, image, sensor_index, from_level=0):
        p = self._build_pyramid(image, from_level)
        ref_p = self.ref_pyramids[sensor_index]

        losses = torch.zeros(len(ref_p)-from_level)
        for l in range(len(p)):
            losses[l] = self.criterion(ref_p[l+from_level], p[l])

        if self._reweight:
            with torch.no_grad():
                normalization = self._reduction(losses)

            for l in range(len(p)):
                losses[l] /= losses[l].item()

            losses *= normalization
                
        loss = self._reduction(losses)
        return loss, losses

#####################################################
## VTK volume writer
######################################################

class VtkWriter():
    def __init__(self, path, dim, scale):
        os.makedirs(path[:path.rfind("/")], exist_ok=True)
        self.path = path
        self.vtk_group = VtkGroup(path)
        self.spacing = scale / (dim - 1)

    def record_epoch(self, epoch, sdf, grad):
        out_fn = f"{self.path}_{epoch}"
        imageToVTK(out_fn, [0]*3, [self.spacing]*3, pointData={"sdf" : sdf, "grad": grad})
        self.vtk_group.addFile(filepath=out_fn + ".vti", sim_time=epoch)
