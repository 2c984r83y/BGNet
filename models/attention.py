import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_2d(nn.Module):
    def __init__(self, input_channel, head_number):
        super(Attention_2d, self).__init__()
        self.head_number = head_number
        self.input_channel = input_channel
        self.head_dim = input_channel // head_number

        self.query_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)
        self.key_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)
        self.value_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)

        self.out_linear = nn.Conv2d(input_channel, input_channel, kernel_size=1)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Split the input into multiple heads
        query = self.query_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, self.head_number, self.head_dim, height, width).transpose(1, 2)

        # Scale the query
        query = query / (self.head_dim ** (1/4))

        # Compute the attention weights
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply the attention weights to the value
        attn_output = torch.matmul(attn_weights, value)

        # Merge the heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.input_channel, height, width)

        # Apply the output linear layer
        output = self.out_linear(attn_output)
        return output
    

# class Attention_3d(nn.Module):
#     def __init__(self, channels_3d, num_heads=8, block=4):
#         """
#         ws 1 for stand attention
#         """
#         super(Attention_3d, self).__init__()
#         self.block = block
#         self.dim_3d = channels_3d
#         self.num_heads = num_heads
#         head_dim_3d = self.dim_3d // num_heads  # 每个head的维度
#         self.scale_3d = head_dim_3d ** -0.5 # 缩放因子
#         # 全连接层可以将输入数据的特征数从 self.dim_3d 扩大到 self.dim_3d 的3倍
#         # 以计算查询、键和值这三个部分
#         self.qkv_3d = nn.Linear(self.dim_3d, self.dim_3d * 3, bias=True)
#         self.final1x1 = torch.nn.Conv3d(self.dim_3d, self.dim_3d, 1)


#     def forward(self, x):
#         # pad input to be a multiple of block
#         B, C, D, H0, W0 = x.shape
#         print(x.shape)
#         pad_l = pad_t = 0
#         pad_r = (self.block[2] - W0 % self.block[2]) % self.block[2]
#         pad_b = (self.block[1] - H0 % self.block[1]) % self.block[1]
#         # pad zeros at edge to make the input shape to be a multiple of block
#         x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
#         B, C, D, H, W = x.shape
#         print(x.shape)
#         d, h, w = D//self.block[0], H//self.block[1], W//self.block[2]
#         # x: [B, C, D, H, W] -> [B, d, h, w, self.block[0], self.block[1], self.block[2], C]
#         x = x.view(B, C, d,self.block[0], h, self.block[1], w, self.block[2]).permute(0, 2, 4, 6, 3, 5, 7, 1)
#         # self.qkv_3d(x) 线性变换，将输入数据的特征数从 self.dim_3d 扩大到 self.dim_3d 的3倍
#         # C // self.num_heads 是每个注意力头的通道数
#         # (3, B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads)
#         # reshape: tensor 一维展开后再重新组织
#         # permute: tensor 维度换位
#         qkv_3d = self.qkv_3d(x).reshape(B, d*h*w, self.block[0]*self.block[1]*self.block[2], 3, self.num_heads,
#                                             C // self.num_heads).permute(3, 0, 1, 4, 2, 5)
#         # assign q, k, v
#         # (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads)
#         q_3d, k_3d, v_3d = qkv_3d[0], qkv_3d[1], qkv_3d[2]
#         # 计算Query(q_3d) 和 Key(k_3d)相似性
#         # @: torch.matmul
#         # 要求第一个矩阵的列数（最后一个维度）等于第二个矩阵的行数（倒数第二个维度）
#         # a: (2, 3, 4) , b: (4, 5)
#         # (2, 3, 4) @ (4, 5) -> (2, 3, 5)
#         # 使用广播机制将 b 扩充为 (2, 4, 5)
#         # 对每个(3, 4) 和 (4, 5)做矩阵乘法，得到(3, 5)的矩阵
#         # 然后把两个(3, 5)的矩阵拼接起来，得到(2, 3, 5)的矩阵
#         # (a,b,c,d,e,f) @ (a,b,c,d,f,e) -> (a,b,c,d,e,e)
#         # 转置最后和倒数第二个维度后进行矩阵乘法，得到注意力权重系数
#         # q_3d: (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], C // self.numheads) 
#         # k_3d: (B, d*h*w, self.numheads, C // self.numheads, self.block[0]*self.block[1]*self.block[2]) 
#         # attn: (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], self.block[0]*self.block[1]*self.block[2])
#         attn = (q_3d @ k_3d.transpose(-2, -1)) * self.scale_3d
#         if pad_r > 0 or pad_b > 0:
#             mask = torch.zeros((1, H, W), device=x.device)
#             # 最后 pad_b 行和所有的列填充为 1
#             mask[:, -pad_b:, :].fill_(1)
#             # 所有的行的最后 pad_r 列填充为 1
#             mask[:, :, -pad_r:].fill_(1)
#             mask = mask.reshape(1, h, self.block[1], w, self.block[2]).transpose(2, 3).reshape(1,  h*w, self.block[1]*self.block[2])
#             # unsqueeze: 在指定位置增加一个维度
#             attn_mask = mask.unsqueeze(2) - mask.unsqueeze(3)  # 1, _h*_w, self.block*self.block, self.block*self.block
#             #  -1000.0 在 softmax 操作后的值接近于 0, 有效地屏蔽掉某些位置
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-1000.0)).masked_fill(attn_mask == 0, float(0.0))
#             # 在第二维上重复 d 次, 在第三维上重复 self.block[0] 次, 在第四维上重复 self.block[0] 次
#             attn = attn + attn_mask.repeat(1, d, self.block[0], self.block[0]).unsqueeze(2)
#         # 对 atten 进行归一化处理
#         attn = torch.softmax(attn, dim=-1)
#         # 根据权重系数对 Value(v_3d) 进行加权求和
#         # 将 attn 张量的每一行（最后一个维度，因为 dim=-1）
#         # 转换为一个概率分布，每一行的所有元素的和都为 1
#         # x -> (B, d*h*w, self.numheads, self.block[0]*self.block[1]*self.block[2], self.block[0]*self.block[1]*self.block[2], C // self.numheads)
#         # x -> (B, d, h ,w, self.num_heads, self.block[0], self.block[1], self.block[2], -1)
#         # x -> (B, self.num_heads, -1, d, self.block[0], h, self.block[1], w, self.block[2])
#         # tensor 中的 -1 表示自动计算该维度的大小
#         x = (attn @ v_3d).view(B, d, h ,w, self.num_heads, self.block[0], self.block[1], self.block[2], -1).permute(0,4,8,1,5,2,6,3,7)
#         # 注意到: d, h, w = D//self.block[0], H//self.block[1], W//self.block[2]
#         x = x.reshape(B, C, D, H, W)
#         if pad_r > 0 or pad_b > 0:
#             x = x[:, :, :, :H0, :W0]
#         return self.final1x1(x)


class Attention_3d(nn.Module):
    def __init__(self, input_channel, head_number):
        super(Attention_3d, self).__init__()
        self.head_number = head_number
        self.input_channel = input_channel
        self.head_dim = input_channel // head_number

        self.query_linear = nn.Conv3d(input_channel, input_channel, kernel_size=1)
        self.key_linear = nn.Conv3d(input_channel, input_channel, kernel_size=1)
        self.value_linear = nn.Conv3d(input_channel, input_channel, kernel_size=1)

        self.out_linear = nn.Conv3d(input_channel, input_channel, kernel_size=1)

    def forward(self, x):
        batch_size, _, depth, width, height = x.size()

        # Split the input into multiple heads
        query = self.query_linear(x).view(batch_size, self.head_number, self.head_dim, depth, width, height).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, self.head_number, self.head_dim, depth, width, height).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, self.head_number, self.head_dim, depth, width, height).transpose(1, 2)

        # Scale the query
        query = query / (self.head_dim ** (1/4))

        # Compute the attention weights
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply the attention weights to the value
        attn_output = torch.matmul(attn_weights, value)

        # Merge the heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.input_channel, depth, width, height)

        # Apply the output linear layer
        output = self.out_linear(attn_output)

        return output
