import torch.nn as nn
import torch.nn.functional as F

class UNet2D(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2D, self).__init__()
        
        self.first = First2D(in_channels, 64, 64)
        self.encoder2d_1 = Encoder2D(64, 128, 128)
        self.encoder2d_2 = Encoder2D(128, 256, 256)
        self.encoder2d_3 = Encoder2D(256, 512, 512)

        self.center = Center2D(512, 1024, 1024, 512)

        self.decoder2d_1 = Decoder2D(1024, 512, 512, 256)
        self.decoder2d_2 = Decoder2D(512, 256, 256, 128)
        self.decoder2d_3 = Decoder2D(256, 128, 128, 64)

        # FIXED: Adjusted in_channels to 256 to match concatenated input (128+128)
        self.last = Last2D(128, 64, out_channels)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x0 = self.first(x)
        x1 = self.encoder2d_1(x0)
        x2 = self.encoder2d_2(x1)
        x3 = self.encoder2d_3(x2)

        c = self.center(x3)
        cat1 = torch.cat([pad_to_shape(c, x3.shape), x3], dim=1)
        d1 = self.decoder2d_1(cat1)

        cat2 = torch.cat([pad_to_shape(d1, x2.shape), x2], dim=1)
        d2 = self.decoder2d_2(cat2)

        cat3 = torch.cat([pad_to_shape(d2, x1.shape), x1], dim=1)
        d3 = self.decoder2d_3(cat3)

        cat4 = torch.cat([pad_to_shape(d3, x0.shape), x0], dim=1)
        out = self.last(cat4)

        return self.activation(out)


class First2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First2D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        self.first = nn.Sequential(*layers)
    def forward(self, x):
        return self.first(x)

class Encoder2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False, downsample_kernel=2):
        super(Encoder2D, self).__init__()
        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)

class Center2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center2D, self).__init__()
        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        self.center = nn.Sequential(*layers)
    def forward(self, x):
        return self.center(x)

class Decoder2D(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder2D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]
        if dropout:
            layers.append(nn.Dropout2d(p=dropout))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder(x)

class Last2D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last2D, self).__init__()
        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
        ]
        if softmax:
            layers.append(nn.Softmax(dim=1))
        self.first = nn.Sequential(*layers)
    def forward(self, x):
        return self.first(x)


def pad_to_shape(tensor, target_shape):
    
    _, _, h, w = tensor.shape
    _, _, ht, wt = target_shape
    pad_h = ht - h
    pad_w = wt - w
    pad = (0, pad_w, 0, pad_h)
    return F.pad(tensor, pad)
