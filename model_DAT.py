import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange
import numbers

def PhiTPhi_fun(x,y, PhiW):
    temp = F.conv2d(x, PhiW, padding=0,stride=32, bias=None)
    temp = temp-y
    temp = F.conv_transpose2d(temp, PhiW, stride=32)
    return temp

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# Define Cross Attention Block
class CrAttention(torch.nn.Module):
    def __init__(self, channels):
        super(CrAttention, self).__init__()
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        
        self.norm_x = LayerNorm(1, 'WithBias')
        self.norm_z = LayerNorm(31, 'WithBias')

        self.t = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )
        self.p = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.channels, kernel_size=1, stride=1, bias=True),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, groups=self.channels, bias=True)
        )

        self.w = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1, bias=True)

        self.pos_emb = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
            nn.GELU(),
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False, groups=self.channels),
        )

    def forward(self, x, z):
        
        x0 = self.norm_x(x)
        z0 = self.norm_z(z)
        
        z1 = self.t(z0)
        b, c, h, w = z1.shape
        

        z1 = z1.view(b, c, -1) # b, c, hw
        x1 = self.p(x0) # b, c, hw
        x1 = x1.view(b, c, -1)
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        x1 = torch.nn.functional.normalize(x1, dim=-1)
        x_t = x1.permute(0, 2, 1) # b, hw, c
        att = torch.matmul(z1, x_t)
        att = self.softmax(att) # b, c, c
        
        z2 = self.t(z0)
        z_v = z2.view(b, c, -1)
        out_x = torch.matmul(att, z_v)
        out_x = out_x.view(b, c, h, w)
        out_x = self.w(out_x) + z +self.pos_emb(z2)


        return out_x

    

    
# Define BasicBlock
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.cra = CrAttention(channels=31)
        self.norm1 = LayerNorm(32, 'WithBias')
        self.norm2 = LayerNorm(32, 'WithBias')
        self.conv_forward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        self.conv_backward = nn.Sequential(
            nn.Conv2d(32, 32 * 4, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32 * 4, 3, 1, 1, bias=False, groups=32 * 4),
            nn.GELU(),
            nn.Conv2d(32 * 4, 32, 1, 1, bias=False),
        )
        
        self.v = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, bias=True)
        self.w = nn.Sequential(
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1, bias=False, groups=31),
            nn.GELU(),
            nn.Conv2d(31, 31, kernel_size=3, stride=1, padding=1, bias=False, groups=31),
        )

        
    def forward(self, x, z_pre, z_cur, PhiWeight, y):
        

        

        z_=self.w(z_cur-z_pre)


        x = x - self.lambda_step *(PhiTPhi_fun(x,y, PhiWeight))

        x_input = self.cra(x, z_)
        x_input = self.v(torch.cat([x, x_input], 1))
        
        x = self.norm1(x_input)
        x_forward = self.conv_forward(x) + x_input
        x = self.norm2(x_forward)
        x_backward = self.conv_backward(x) + x_forward

        x_dual=x_backward
        
        
        x = x_dual[:, :1, :, :]
        
        res=PhiTPhi_fun(x,y, PhiWeight)
        
        z_c=x_dual[:, 1:, :, :]
        
        z_pre = z_cur

        z_cur=self.cra(res,z_c)+z_cur

        return x,z_pre,z_cur

# Define DAT
class DAT(torch.nn.Module):
    def __init__(self, LayerNo, sensing_rate):
        super(DAT, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        self.patch_size = 32
        self.n_input = int(sensing_rate * 1024)

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.Phiweight = nn.Parameter(init.xavier_normal_(torch.Tensor(self.n_input, 1, self.patch_size, self.patch_size)))
        self.fcs = nn.ModuleList(onelayer)
        self.fe = nn.Conv2d(1, 31, 3, padding=1, bias=True)
        self.fe2 = nn.Conv2d(1, 31, 3, padding=1, bias=True)

    def forward(self, x):

        y = F.conv2d(x, self.Phiweight, stride=self.patch_size, padding=0, bias=None) # 64*1089*3*3 

        x_0 = F.conv_transpose2d(y, self.Phiweight, stride=self.patch_size) 
        x = x_0
        z_pre = self.fe(x)
        z_cur = self.fe2(x)


        for i in range(self.LayerNo):
            [x,z_pre,z_cur]= self.fcs[i](x, z_pre, z_cur,self.Phiweight, y)


        x_final = x

        return x_final
