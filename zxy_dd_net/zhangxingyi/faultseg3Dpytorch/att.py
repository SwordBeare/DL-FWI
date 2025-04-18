import torch
import torch.nn as nn


def maxpool2x2(x):
    mp = nn.MaxPool3d(kernel_size=2, stride=2)
    x = mp(x)
    return x


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.encoder_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
            # 原论文中是每层有两个卷积核进行下采样 https://arxiv.org/abs/1505.04597
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.encoder_block(x)
        return x


class CenterBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(CenterBlock, self).__init__()
        mid_channels = int(in_channels * 2)

        self.center_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(in_channels * 2), mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.center_block(x)
        return x


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        mid_channels = int(in_channels / 2)

        self.decoder_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(mid_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.decoder_block(x)
        return x


class FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FinalBlock, self).__init__()
        mid_channels = int(in_channels / 2)

        self.final_block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, 1, bias=False),
        )

    def forward(self, x):
        x = self.final_block(x)
        return x


class AttentionGates3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGates3D, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out





class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Encoder Part
        self.encoder1 = EncoderBlock(in_channels=in_channels, out_channels=32)
        self.encoder2 = EncoderBlock(in_channels=32, out_channels=64)
        self.encoder3 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder4 = EncoderBlock(in_channels=128, out_channels=256)
        # Center part.
        self.center = CenterBlock(in_channels=256, out_channels=256)
        # Decoder part.
        self.decoder4 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=32)
        # Final part.
        self.final = FinalBlock(in_channels=64, out_channels=out_channels)
        # Attention Gate
        self.att4 = AttentionGates3D(F_g=256, F_l=256, F_int=128)
        self.att3 = AttentionGates3D(F_g=128, F_l=128, F_int=64)
        self.att2 = AttentionGates3D(F_g=64, F_l=64, F_int=32)
        self.att1 = AttentionGates3D(F_g=32, F_l=32, F_int=16)

    def forward(self, x):
        # Encoding, compressive pathway.
        out_encoder1 = self.encoder1(x)
        out_endocer1_mp = maxpool2x2(out_encoder1)
        out_encoder2 = self.encoder2(out_endocer1_mp)
        out_endocer2_mp = maxpool2x2(out_encoder2)
        out_encoder3 = self.encoder3(out_endocer2_mp)
        out_endocer3_mp = maxpool2x2(out_encoder3)
        out_encoder4 = self.encoder4(out_endocer3_mp)
        # Decoding, expansive pathway.
        out_endocer4_mp = maxpool2x2(out_encoder4)
        out_center = self.center(out_endocer4_mp)
        out_att4 = self.att4(g=out_center, x=out_encoder4)
        out_decoder4 = self.decoder4(torch.cat((out_center, out_att4), 1))
        out_att3 = self.att3(g=out_decoder4, x=out_encoder3)
        out_decoder3 = self.decoder3(torch.cat((out_decoder4, out_att3), 1))
        out_att2 = self.att2(g=out_decoder3, x=out_encoder2)
        out_decoder2 = self.decoder2(torch.cat((out_decoder3, out_att2), 1))
        out_att1 = self.att1(g=out_decoder2, x=out_encoder1)
        out_final = self.final(torch.cat((out_decoder2, out_att1), 1))

        return out_final


# if __name__ == '__main__':
#     data_ = torch.rand(1, 1, 64, 64, 64)
#     open_fwi_data = AttentionUNet3D(in_channels=1, out_channels=2)
#     print(open_fwi_data)
#     out_data = open_fwi_data(data_)
#     print(out_data.shape)
#
#     print("*" * 30 + "\n |\t\tEnd Of Program\t\t|\n" + "*" * 30)
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image_size = 128
#     x = torch.Tensor(1, 1, image_size, image_size, image_size)
#     x.to(device)
#     print("x size: {}".format(x.size()))
#
#     open_fwi_data = UNet(in_channels=1, out_channels=1)
#
#     out = open_fwi_data(x)
#     print("out size: {}".format(out.size()))
