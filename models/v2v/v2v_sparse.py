# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

def padding(x, num):

    return 

class Basic3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic3DBlock, self).__init__()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=1,
                dimension=3),
            ME.MinkowskiBatchNorm(out_planes),
            # nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            # nn.BatchNorm3d(out_planes),
            # nn.ReLU(True)
        )
    
    def forward(self, x):
        return MF.relu(self.block(x))

class Res3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res3DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=1,
                dimension=3),
            ME.MinkowskiBatchNorm(out_planes),
            ME.MinkowskiConvolution(in_channels=out_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=1,
                dimension=3),
            ME.MinkowskiBatchNorm(out_planes)
            # nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes),
            # nn.ReLU(True),
            # nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    stride=1,
                    dimension=3),
                ME.MinkowskiBatchNorm(out_planes)
                # nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                # nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        # import pdb; pdb.set_trace()
        skip = self.skip_con(x)
        res = self.res_branch(x)
        res += skip
        return MF.relu(res)


class Pool3DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3DBlock, self).__init__()
        self.pool_size = pool_size
        self.pool_bolck = ME.MinkowskiMaxPooling(kernel_size=3, stride=2,
                    dimension=3)
    def forward(self, x):
        # return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)
        return self.pool_bolck(x)
    

class Upsample3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample3DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    dimension=3),
            ME.MinkowskiBatchNorm(out_planes)
            # nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            # nn.BatchNorm3d(out_planes),
            # nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)
    

class EncoderDecorder(nn.Module):
    def __init__(self):
        super(EncoderDecorder, self).__init__()

        self.encoder_pool1 = Pool3DBlock(2)
        self.encoder_res1 = Res3DBlock(32, 64)
        self.encoder_pool2 = Pool3DBlock(2)
        self.encoder_res2 = Res3DBlock(64, 128)

        self.mid_res = Res3DBlock(128, 128)

        self.decoder_res2 = Res3DBlock(128, 128)
        self.decoder_upsample2 = Upsample3DBlock(128, 64, 2, 2)
        self.decoder_res1 = Res3DBlock(64, 64)
        self.decoder_upsample1 = Upsample3DBlock(64, 32, 2, 2)

        self.skip_res1 = Res3DBlock(32, 32)
        self.skip_res2 = Res3DBlock(64, 64)

    def forward(self, x):
        skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)

        skip_x2 = self.skip_res2(x)
        # import pdb; pdb.set_trace()
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)

        x = self.mid_res(x)

        x = self.decoder_res2(x)
        x = self.decoder_upsample2(x)
        x = x + skip_x2

        x = self.decoder_res1(x)
        x = self.decoder_upsample1(x)
        x = x + skip_x1

        return x


# class EncoderDecorder(nn.Module):
#     def __init__(self):
#         super(EncoderDecorder, self).__init__()

#         self.encoder_pool1 = Pool3DBlock(2)
#         self.encoder_res1 = Res3DBlock(16, 32)
#         self.encoder_pool2 = Pool3DBlock(2)
#         self.encoder_res2 = Res3DBlock(32, 64)

#         self.mid_res = Res3DBlock(64, 64)

#         self.decoder_res2 = Res3DBlock(64, 64)
#         self.decoder_upsample2 = Upsample3DBlock(64, 32, 2, 2)
#         self.decoder_res1 = Res3DBlock(32, 32)
#         self.decoder_upsample1 = Upsample3DBlock(32, 16, 2, 2)

#         self.skip_res1 = Res3DBlock(16, 16)
#         self.skip_res2 = Res3DBlock(32, 32)

#     def forward(self, x):
#         skip_x1 = self.skip_res1(x)
#         x = self.encoder_pool1(x)
#         x = self.encoder_res1(x)

#         skip_x2 = self.skip_res2(x)
#         x = self.encoder_pool2(x)
#         x = self.encoder_res2(x)

#         x = self.mid_res(x)

#         x = self.decoder_res2(x)
#         x = self.decoder_upsample2(x)
#         x = x + skip_x2

#         x = self.decoder_res1(x)
#         x = self.decoder_upsample1(x)
#         x = x + skip_x1

#         return x

class V2VNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(V2VNet, self).__init__()

        self.front_layers = nn.Sequential(
            Basic3DBlock(input_channels, 16, 7),
            Res3DBlock(16, 32),
        )

        self.encoder_decoder = EncoderDecorder()

        # self.output_layer = nn.Conv3d(32, output_channels, kernel_size=1, stride=1, padding=0)
        self.output_layer = ME.MinkowskiConvolutionTranspose(in_channels=32,
                    out_channels=output_channels,
                    kernel_size=1,
                    stride=1,
                    dimension=3)

        self.weight_initialization()

    def forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        x = self.output_layer(x)

        return x

    def feat_forward(self, x):
        x = self.front_layers(x)
        x = self.encoder_decoder(x)
        return x

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
