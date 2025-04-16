# 3D-UNet open_fwi_data.
# x: 128x128 resolution for 32 frames.
import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)


class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):  # 1 1 4
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.ReLU()

        self.down_1 = conv_block_3d(self.in_dim, self.num_filters, activation)
        self.Dropout1=nn.Dropout()
        self.down_2 = conv_block_3d(self.num_filters, self.num_filters*2, activation)
        self.trans_0 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        #self.down_3 = conv_block_3d(self.num_filters * 2, self.num_filters*4, activation)
        #self.Dropout2=nn.Dropout()
        self.down_4 = conv_block_3d(self.num_filters*2, self.num_filters*4, activation)
        self.pool_0 = max_pooling_3d()
        #concat_1 = torch.cat([pool_1, down_2], dim=1)
        self.down_5 = conv_block_3d(self.num_filters*6, out_dim, activation)
        self.out1=nn.Dropout()



        ##############
        #  后置网络  ##
        ##############

        # self.in_dim = in_dim
        # self.out_dim = out_dim
        self.num_filters = num_filters
        activation1 = nn.LeakyReLU(0.2, inplace=True)  # 该函数相比于ReLU，保留了一些负轴的值，缓解了激活值过小而导致神经元参数无法更新的问题

        # 假设输入图片为 W x W 卷积核大小为FxF，步长stride=S，padding设置为P（填充的像素数）
        # 则输出图像的大小=（W - F +2P）/S +1

        # Down sampling
        self.down_6 = conv_block_2_3d(out_dim, self.num_filters, activation1)
        self.pool_1 = max_pooling_3d()
        self.down_7 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation1)
        self.pool_2 = max_pooling_3d()
        self.down_8 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation1)
        self.pool_3 = max_pooling_3d()
        self.down_9 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation1)
        self.pool_4 = max_pooling_3d()
        self.down_10 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation1)
        self.pool_5 = max_pooling_3d()

        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation1)

        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation1)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation1)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation1)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation1)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation1)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation1)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation1)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation1)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation1)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation1)

        # Output
        self.last_relu = activation1
        self.out = conv_block_3d(self.num_filters, out_dim, self.last_relu)



    def forward(self, x):
        down_1 = self.down_1(x)
        Dropout1=self.Dropout1(down_1)
        down_2=self.down_2(Dropout1)

        trans_0=self.trans_0(down_2)
        #down_3=self.down_3(trans_0)
        #Dropout2 = self.Dropout2(down_3)
        down_4 = self.down_4(trans_0)
        pool_0=self.pool_0(down_4)

        concat_0 = torch.cat([pool_0, down_2], dim=1)
        down_5 = self.down_5(concat_0)

        out1 = self.out1(down_5)

        ############
        #  后置网络  #
        ############

        down_6 = self.down_6(out1)  # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_6)  # -> [1, 4, 64, 64, 64]

        down_7 = self.down_7(pool_1)  # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_7)  # -> [1, 8, 32, 32, 32]

        down_8 = self.down_8(pool_2)  # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_8)  # -> [1, 16, 16, 16, 16]

        down_9 = self.down_9(pool_3)  # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_9)  # -> [1, 32, 8, 8, 8]

        down_10 = self.down_10(pool_4)  # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_10)  # -> [1, 64, 4, 4, 4]

        # Bridge
        bridge = self.bridge(pool_5)  # -> [1, 128, 4, 4, 4]

        # Up sampling
        trans_1 = self.trans_1(bridge)  # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_10], dim=1)  # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1)  # -> [1, 64, 8, 8, 8]

        trans_2 = self.trans_2(up_1)  # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_9], dim=1)  # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2)  # -> [1, 32, 16, 16, 16]

        trans_3 = self.trans_3(up_2)  # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_8], dim=1)  # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3)  # -> [1, 16, 32, 32, 32]

        trans_4 = self.trans_4(up_3)  # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_7], dim=1)  # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)  # -> [1, 8, 64, 64, 64]

        trans_5 = self.trans_5(up_4)  # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_6], dim=1)  # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5)  # -> [1, 4, 128, 128, 128]

        # Output
        out = self.out(up_5)  # -> [1, 1, 128, 128, 128]
        # out = self.last_relu(out)
        return out




class UNet1(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):  # 1 1 16
        super(UNet1, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True) #该函数相比于ReLU，保留了一些负轴的值，缓解了激活值过小而导致神经元参数无法更新的问题

        #假设输入图片为 W x W 卷积核大小为FxF，步长stride=S，padding设置为P（填充的像素数）
        #则输出图像的大小=（W - F +2P）/S +1

        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.last_relu = activation
        self.out = conv_block_3d(self.num_filters, out_dim, self.last_relu)
        
    
    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        
        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
        
        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        
        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
        
        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 1, 128, 128, 128]
        # out = self.last_relu(out)
        return out

# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     image_size = 128
#     x = torch.Tensor(1, 1, image_size, image_size, image_size)
#     x.to(device)
#     print("x size: {}".format(x.size()))
# # with torch.no_grad():
#     open_fwi_data = UNet(in_dim=1, out_dim=1, num_filters=4)
#     #open_fwi_data.eval()
#
#     out = open_fwi_data(x)
#     print("out size: {}".format(out.size()))
