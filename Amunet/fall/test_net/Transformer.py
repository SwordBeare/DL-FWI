import argparse
import copy
import math
import torch
import torch.nn as nn

# from .Unet import *

# 完整的Transformer过程包括Embedding、Encoder和Decoder
class Transformer(nn.Module):
    def __init__(self, config, n_patch, is_open_FWI=False):
        super().__init__()
        self.hidden_size = config.patch_size[0] * config.patch_size[1] * config.down_sample_list[-1]
        # Embedding过程
        self.embeddings = Embedding(config, n_patch)
        # Encoder过程
        self.encoder = Encoder(config)
        # Decoder过程
        self.decoder = Decoder(config, n_patch)

    def forward(self, x, features = None):
        embedding_out = self.embeddings(x)
        encoded = self.encoder(embedding_out)
        decoded, features = self.decoder(encoded, features)
        return decoded, features

# Embedding过程是将(B, 512, 13, 10)卷积成(B, 2048, 6, 5)，并通过扁平、转置等操作，变成(B, 30, 2048)
# 把原本的(C,H,W)图片数据处理成(H/2 * W/2, 2*2*C)的序列数据
# Hidden_size的含义是：在transformer过程中序列数据的长度
class Embedding(nn.Module):
    def __init__(self, config, n_patche):
        super().__init__()
        # 将(B, 512, 13, 10)卷积成(B, 2048, 6, 5)
        self.patch_embeddings = nn.Conv2d(config.down_sample_list[-1],
                                          config.hidden_size,
                                          kernel_size=config.patch_size,
                                          stride=config.patch_size)
        # 生成待嵌入的位置信息（这里的1是固定的吗？若batch_size不为1会出错吗？）
        # self.position_embeddings = nn.Parameter(torch.zeros(1, 475, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patche[0] * n_patche[1], config.hidden_size))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.patch_embeddings(x)
        # 扁平化，由[batch_size, height, width, channels]变为[batch_size, height*width*channels]
        x = x.flatten(2)
        # 转置
        x = x.transpose(-1, -2)
        # 将原图片与位置信息相加
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

# Encoder过程包含了若干个transformer层，将其封装在Block()方法中
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        # 添加多个连续的transformer层
        for _ in range(config.num_transformer_layer):
            layer_ = Block(config)
            self.layer.append(copy.deepcopy(layer_))

    def forward(self, hidden_states):
        # 连续经过多个transformer层
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)  # (B, n_patch, hidden)
        # 经过批归一化后再输出
        encoded = self.encoder_norm(hidden_states)
        return encoded

# transformer层包括Attention多头自注意力机制、残差连接和Mlp前馈网络
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        # 封装好的多头自注意力机制
        self.attn = Attention(config)
        # 封装好的前馈网络
        self.ffn = Mlp(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x

# 多头自注意力机制，用于计算输入的注意力权重，并生成一个带有编码信息的输出向量，指示中的每个部分如何关注其他所有部分
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 多头数量，单个头的长度
        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # query、key和value是通过线性层得来的，尺寸未发生变化
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # 线性层得到最终输出
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        # 输出前需要归一化
        self.softmax = nn.Softmax(dim=-1)

    # 将长度为Hidden_size的序列分开，将“单头”变为“多头”（具体原理还没有研究）
    def transpose_for_scores(self, x):
        # new_x_shape (B, n_patch, num_attention_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # view()方法主要用于Tensor维度的重构，即返回一个有相同数据但不同维度的Tensor
        x = x.view(*new_x_shape)
        # permute可以对任意高维矩阵进行转置，transpose只能操作2D矩阵的转置
        return x.permute(0, 2, 1, 3)  # return (B, num_attention_heads, n_patch, attention_head_size)

    def forward(self, hidden_states):
        # 初步得到的query、key和value，是单头的
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        # 经过处理，变成多头
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # torch.matmul矩阵相乘
        # key_layer.transpose(-1, -2): (B, num_attention_heads, attention_head_size, n_patch)
        # attention_scores: (B, num_attention_heads, n_patch, n_patch)
        ###### 矩阵相乘得到权重矩阵，并归一化处理
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        # weights = attention_probs if self.vis else None
        # context_layer (B, num_attention_heads, n_patch, attention_head_size)
        ###### 权重矩阵与value相乘，生成一个带有编码信息的输出向量
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer (B, n_patch, num_attention_heads, attention_head_size)
        # contiguous一般与transpose，permute，view搭配使用：使用transpose或permute进行维度变换后，调用contiguous，然后方可使用view对维度进行变形
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape (B, n_patch,all_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # attention_output (B, n_patch,hidden_size)
        # 小细节 attention_head_size = int(hidden_size / num_attention_heads)，all_head_size = num_attention_heads * attention_head_size
        # 所以应该满足hidden_size能被num_attention_heads整除
        attention_output = self.out(context_layer)
        return attention_output

# 前馈网络由几个线性层，激活函数组成。前馈网络用于进一步处理注意力输出，可能使其由更丰富的表达。
class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        # nn.init.xavier_uniform_初始化权重,避免深度神经网络训练过程中的梯度消失和梯度爆炸问题
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.normal_是正态初始化函数
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# Decoder负责将上层的序列输出(B, 30, 2048)变成图片形式(B, 1024, 6, 5)
class Decoder(nn.Module):
    def __init__(self, config, n_patch):
        super().__init__()
        self.config = config
        self.n_patch = n_patch
        # 通过一次卷积，将transformer部分的encoder输出的通道数2048变为1024
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            config.decoder_list[0],
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, hidden_states, features):
        B, n_patch, hidden = hidden_states.size()  # hidden_states: (B, n_patch, hidden)
        x = hidden_states.permute(0, 2, 1)  # x: (B, hidden, n_patch)
        x = x.contiguous().view(B, hidden, self.n_patch[0], self.n_patch[1])  # x: (B, hidden, h, w)
        # 变成最终的图片形式(B, 1024, 6, 5)
        x = self.conv_more(x)
        return x, features

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


# 封装了一次卷积操作，包括卷积+ReLU激活+归一化
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


if __name__ == '__main__':
    DownSamplingList = [32,64,128,256,512]
    PatchSize = [2,2]
    Inchannels = 29  # Number of input channels, i.e. the number of shots
    DataDim = [400, 301]  # Dimension of original one-shot seismic data
    ModelDim = [201, 301]  # Dimension of one velocity model
    DecoderList = [1024, 512, 256, 128, 64, 32]
    HiddenSize = PatchSize[0] * PatchSize[1] * DownSamplingList[-1]
    MLPDim = 4 * 128
    NumHeads = 8
    NumTransformerLayer = 4
    parser = argparse.ArgumentParser()


    parser.add_argument('--down_sample_list', type=list,
                        default=DownSamplingList, help='downsampling node list')
    parser.add_argument('--decoder_list', type=list,
                        default=DecoderList, help='decoder node list')
    parser.add_argument('--image_size', type=list,
                        default=DataDim, help='image_size')
    parser.add_argument('--label_size', type=list,
                        default=ModelDim, help='label_size')
    parser.add_argument('--patch_size', type=list,
                        default=PatchSize, help='patch_size')
    parser.add_argument('--in_channel', type=int,
                        default=Inchannels, help='input channel number')
    parser.add_argument('--hidden_size', type=int,
                        default=HiddenSize, help='size of the hidden layer')
    parser.add_argument('--mlp_dim', type=int,
                        default=MLPDim, help='dim of the mlp')
    parser.add_argument('--num_heads', type=int,
                        default=NumHeads, help='number of transformer heads')
    parser.add_argument('--num_transformer_layer', type=int,
                        default=NumTransformerLayer, help='number of the transformer layer')
    args = parser.parse_args()


    model = Transformer(args,n_patch)

