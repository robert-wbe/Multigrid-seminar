import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

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

def reflection_pad(f: torch.Tensor) -> torch.Tensor:
    return F.pad(f.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')[0][0]

def laplace_smoothen(f: torch.Tensor) -> torch.Tensor:
    return conv(reflection_pad(f), neighbor_kernel)


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