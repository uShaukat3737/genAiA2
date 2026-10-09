"""CycleGAN networks as described in Zhu et al. 2017 (Appendix 7.2)."""
import torch.nn as nn

INIT_STD = 0.02


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """c7s1-64, d128, d256, R256 x n_blocks, u128, u64, c7s1-3 (paper uses 9 blocks at 256px)."""

    def __init__(self, channels=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(channels, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        dim = ngf
        for _ in range(2):
            layers += [nn.Conv2d(dim, dim * 2, 3, 2, 1), nn.InstanceNorm2d(dim * 2), nn.ReLU(True)]
            dim *= 2
        layers += [ResnetBlock(dim) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(dim, dim // 2, 3, 2, 1, output_padding=1),
                       nn.InstanceNorm2d(dim // 2), nn.ReLU(True)]
            dim //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, channels, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN: C64 (no norm), C128, C256, C512, then a 1-channel conv."""

    def __init__(self, channels=3, ndf=64):
        super().__init__()
        layers = [nn.Conv2d(channels, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        dim = ndf
        for stride in (2, 2, 1):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride, 1), nn.InstanceNorm2d(dim * 2), nn.LeakyReLU(0.2, True)]
            dim *= 2
        layers += [nn.Conv2d(dim, 1, 4, 1, 1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, INIT_STD)
            nn.init.zeros_(m.bias)
    return net
