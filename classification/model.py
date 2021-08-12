import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Dropout
from timm.models.efficientnet import *
from timm.models.nfnet import *
from collections import OrderedDict
#from efficientunet import *
from efficientnet_pytorch import EfficientNet

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        e = tf_efficientnet_b7(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)

        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]
        self.b8 = nn.Sequential(
            e.conv_head,  # 384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(2560, 4)

        self.mask0 = nn.Sequential(
            nn.ConvTranspose2d(48, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1),
        )

        self.mask1 = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask2 = nn.Sequential(
            nn.Conv2d(384, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask3 = nn.Sequential(
            nn.Conv2d(640, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask4 = nn.Sequential(
            nn.Conv2d(2560, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )


    # @torch.cuda.amp.autocast()
    def forward(self, x):
        batch_size = len(x)
        x = self.b0(x)  # ; torch.Size([8, 64, 256, 256])
        x = self.b1(x)  # ; torch.Size([8, 32, 256, 256])
        x = self.b2(x)  # ; torch.Size([8, 48, 128, 128])
        mask0 = self.mask0(x)
        x = self.b3(x)  # ; torch.Size([8, 80, 64, 64])
        #mask = self.mask(x)
        x = self.b4(x)  # ; torch.Size([8, 160, 32, 32])
        x = self.b5(x)  # ; torch.Size([8, 224, 32, 32])
        mask1 = self.mask1(x)


        # ------------
        # -------------
        x = self.b6(x)  # ; torch.Size([8, 384, 16, 16])
        mask2 = self.mask2(x)
        x = self.b7(x)  # ; torch.Size([8, 640, 16, 16])
        mask3 = self.mask3(x)

        x = self.b8(x)  # ; torch.Size([8, 2560, 16, 16])
        mask4 = self.mask4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, mask0, mask0, mask1, mask2, mask3, mask4
        #return logit, mask1, mask2, mask3



class Net_v2l(nn.Module):
    def __init__(self,):
        super(Net_v2l, self).__init__()

        e = tf_efficientnetv2_l_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)

        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]

        self.b8 = nn.Sequential(
            e.conv_head,  # 384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(1280, 4)

        self.mask0 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1),
        )

        self.mask01 = nn.Sequential(
            nn.ConvTranspose2d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=2, padding=0, stride=2),
        )

        self.mask1 = nn.Sequential(
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask2 = nn.Sequential(
            nn.Conv2d(384, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask3 = nn.Sequential(
            nn.Conv2d(640, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask4 = nn.Sequential(
            nn.Conv2d(1280, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    # @torch.cuda.amp.autocast()
    def forward(self, x):
        batch_size = len(x)
        x = self.b0(x)  # ; torch.Size([8, 32, 256, 256])
        x = self.b1(x)  # ; torch.Size([8, 32, 256, 256])
        x = self.b2(x)  # ; torch.Size([8, 64, 128, 128])
        #print(x.shape)
        mask0 = self.mask0(x)
        #print(mask0.shape)
        x = self.b3(x)  # ; torch.Size([8, 96, 64, 64])
        #mask01 = self.mask01(x)
        x = self.b4(x)  # ; torch.Size([8, 192, 32, 32])
        mask01 = self.mask01(x)
        x = self.b5(x)  # ; torch.Size([8, 224, 32, 32])
        mask1 = self.mask1(x)
        # ------------
        # -------------
        x = self.b6(x)  # ; torch.Size([8, 384, 16, 16])
        mask2 = self.mask2(x)
        x = self.b7(x)  # ; torch.Size([8, 640, 16, 16])
        mask3 = self.mask3(x)
        x = self.b8(x)  # ; torch.Size([8, 1280, 16, 16])
        mask4 = self.mask4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)

        # x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, mask0, mask01, mask1, mask2, mask3, mask4

class Net_v2m(nn.Module):
    def __init__(self,):
        super(Net_v2m, self).__init__()

        e = tf_efficientnetv2_m_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)

        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]
        self.b7 = e.blocks[6]

        self.b8 = nn.Sequential(
            e.conv_head,  # 384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(1280, 4)

        self.mask0 = nn.Sequential(
            nn.ConvTranspose2d(48, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1),
        )
        self.mask1 = nn.Sequential(
            nn.Conv2d(176, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask2 = nn.Sequential(
            nn.Conv2d(304, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask3 = nn.Sequential(
            nn.Conv2d(512, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask4 = nn.Sequential(
            nn.Conv2d(1280, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )


    # @torch.cuda.amp.autocast()
    def forward(self, x):
        batch_size = len(x)
        x = self.b0(x)  # ; torch.Size([8, 24, 256, 256])
        #print(x.shape)
        x = self.b1(x)  # ; torch.Size([8, 24, 256, 256])
        #print(x.shape)
        x = self.b2(x)  # ; torch.Size([8, 48, 128, 128])
        #print(x.shape)
        mask0 = self.mask0(x)
        x = self.b3(x)  # ; torch.Size([8, 80, 64, 64])

        x = self.b4(x)  # ; torch.Size([8, 160, 32, 32])
        x = self.b5(x)  # ; torch.Size([8, 176, 32, 32])
        mask1 = self.mask1(x)
        # ------------
        # -------------
        x = self.b6(x)  # ; torch.Size([8, 304, 16, 16])
        mask2 = self.mask2(x)
        x = self.b7(x)  # ; torch.Size([8, 512, 16, 16])
        mask3 = self.mask3(x)
        x = self.b8(x)  # ; torch.Size([8, 1280, 16, 16])
        mask4 = self.mask4(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)
        return logit, mask0, mask0, mask1, mask2, mask3, mask4

class Net_v2s(nn.Module):
    def __init__(self,):
        super(Net_v2s, self).__init__()

        e = tf_efficientnetv2_s_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)

        self.b0 = nn.Sequential(
            e.conv_stem,
            e.bn1,
            e.act1,
        )
        self.b1 = e.blocks[0]
        self.b2 = e.blocks[1]
        self.b3 = e.blocks[2]
        self.b4 = e.blocks[3]
        self.b5 = e.blocks[4]
        self.b6 = e.blocks[5]

        self.b7 = nn.Sequential(
            e.conv_head,  # 384, 1536
            e.bn2,
            e.act2,
        )

        self.logit = nn.Linear(1280, 4)

        self.mask0 = nn.Sequential(
            nn.ConvTranspose2d(48, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=3, padding=1),
        )
        self.mask1 = nn.Sequential(
            nn.Conv2d(160, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask2 = nn.Sequential(
            nn.Conv2d(256, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask3 = nn.Sequential(
            nn.Conv2d(1280, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

        self.mask4 = nn.Sequential(
            nn.Conv2d(1280, 224, kernel_size=3, padding=1),
            nn.BatchNorm2d(224),
            nn.ReLU(inplace=True),
            nn.Conv2d(224, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        batch_size = len(x)
        x = self.b0(x)  # ; torch.Size([8, 24, 256, 256])
        x = self.b1(x)  # ; torch.Size([8, 24, 256, 256])
        x = self.b2(x)  # ; torch.Size([8, 48, 128, 128])
        mask0 = self.mask0(x)
        x = self.b3(x)  # ; torch.Size([8, 64, 64, 64])
        x = self.b4(x)  # ; torch.Size([8, 128, 32, 32])
        x = self.b5(x)  # ; torch.Size([8, 160, 32, 32])
        mask1 = self.mask1(x)
        # ------------
        # -------------
        x = self.b6(x)  # ; torch.Size([8, 256, 16, 16])
        mask2 = self.mask2(x)
        x = self.b7(x)  # ; torch.Size([8, 1280, 16, 16])
        mask3 = self.mask3(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        # x = F.dropout(x, 0.5, training=self.training)
        logit = self.logit(x)

        return logit, mask0, mask0, mask1, mask2, mask3, mask3
