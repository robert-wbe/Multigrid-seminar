import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from dataclasses import dataclass

# convolutions
def conv(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return F.conv2d(g.unsqueeze(0), h.unsqueeze(0).unsqueeze(0))[0]

def convT(g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return F.conv_transpose2d(g.unsqueeze(0), h.unsqueeze(0).unsqueeze(0), stride=2, padding=1)[0]

# kernels
interp_kernel = torch.tensor(((0.25, 0.5, 0.25), (0.5, 1., 0.5), (0.25, 0.5, 0.25)))
neighbor_kernel = torch.tensor(((0., 0.25, 0.), (0.25, 0., 0.25), (0., 0.25, 0.)))

# up/downsampling
def interpolate(f: torch.Tensor) -> torch.Tensor:
    return convT(f, interp_kernel)

def restrict(f: torch.Tensor) -> torch.Tensor:
    return f[::2, ::2]

# boundary conditions
@dataclass
class BoundaryCondition:
    mask: torch.Tensor
    vals: torch.Tensor

    def restrict(self) -> BoundaryCondition:
        return BoundaryCondition(restrict(self.mask), restrict(self.vals))
    
    @staticmethod
    def empty_for_grid(grid: torch.Tensor) -> BoundaryCondition:
        return BoundaryCondition(torch.full_like(grid, False), torch.zeros_like(grid))
    
    @staticmethod
    def empty_w_size(size) -> BoundaryCondition:
        return BoundaryCondition(torch.full(size, False), torch.zeros(size))
    
    def __setitem__(self, accessor, value):
        self.mask[accessor] = True
        self.vals[accessor] = value

    def mean(self) -> torch.Tensor:
        return self.vals[self.mask].mean()

@dataclass
class NeumannBoundaryCondition:
    left: torch.Tensor
    right: torch.Tensor
    bottom: torch.Tensor
    top: torch.Tensor

    def restrict(self) -> NeumannBoundaryCondition:
        return NeumannBoundaryCondition(self.left[::2], self.right[::2], self.bottom[::2], self.top[::2])
    
    @staticmethod
    def from_vector_field(v, coord_grid: torch.Tensor) -> NeumannBoundaryCondition:
        return NeumannBoundaryCondition(
            left   = -v(coord_grid[:, 0])[:, 0],
            right  = +v(coord_grid[:, -1])[:, 0],
            bottom = -v(coord_grid[0, :])[:, 1],
            top    = +v(coord_grid[-1, :])[:, 1],
        )

def neumann_nd(bc: NeumannBoundaryCondition) -> torch.Tensor:
    h, w = bc.left.numel(), bc.bottom.numel()
    nd = torch.zeros(h, w)
    nd[0, :] += bc.bottom
    nd[-1, :] += bc.top
    nd[:, 0] += bc.left
    nd[:, -1] += bc.right
    return nd



def reflection_pad(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')[0, 0]

def repl_pad(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')[0, 0]

def laplace_smoothen(f: torch.Tensor) -> torch.Tensor:
    return conv(reflection_pad(f), neighbor_kernel)

def poisson_smoothen(u: torch.Tensor, f: torch.Tensor, bc: NeumannBoundaryCondition, h: float) -> torch.Tensor:
    return conv(repl_pad(u), neighbor_kernel) + (h / 4) * neumann_nd(bc) - (h**2) * f

def open_image(fname: str) -> torch.Tensor:
    img = Image.open(fname)
    return pil_to_tensor(img)

if __name__ == '__main__':
    test = torch.randn(3, 3)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(test, vmin=test.min(), vmax=test.max())
    ax2.imshow(reflection_pad(test), vmin=test.min(), vmax=test.max())
    ax3.imshow(laplace_smoothen(test), vmin=test.min(), vmax=test.max())
    plt.show()