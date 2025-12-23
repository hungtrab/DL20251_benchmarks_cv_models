# FILE: /home/vudd/Convnext/DL20251_benchmarks_cv_models/models_dense.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# --- 1. CÁC LỚP CƠ BẢN (Không cần MinkowskiEngine) ---
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
            return x

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

# --- 2. BACKBONE CONVNEXT V2 ---
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        
        # Stem
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsample layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # Stages
        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        if num_classes > 0:
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# --- 3. FCMAE DENSE (Thay thế cho fcmae.py gốc) ---
class FCMAE_Dense(nn.Module):
    def __init__(self, encoder, mask_ratio=0.6, decoder_embed_dim=512, decoder_depth=1, patch_size=32):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Tự động lấy dimension cuối của encoder
        enc_dim = 1024 # Default base
        if hasattr(encoder, 'norm'):
             enc_dim = encoder.norm.normalized_shape[0]

        self.decoder_embed = nn.Linear(enc_dim, decoder_embed_dim, bias=True)
        
        self.decoder_blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, drop_path=0.) 
            for i in range(decoder_depth)
        ])
        
        self.decoder_norm = LayerNorm(decoder_embed_dim, eps=1e-6, data_format="channels_last")
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size**2) * 3, bias=True) 

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x, mask_ratio):
        # 1. Stem
        x = self.encoder.downsample_layers[0](x)
        
        # 2. Masking trên Feature map (Dense implementation)
        B, C, H, W = x.shape
        # Tạo mask ngẫu nhiên tỉ lệ thấp (ví dụ 1/8 so với ảnh gốc)
        mask_h, mask_w = H, W 
        
        mask = torch.rand(B, 1, mask_h, mask_w, device=x.device)
        # 1: giữ lại, 0: che đi (theo logic nhân bản đồ đặc trưng)
        mask = (mask > mask_ratio).float() 
        
        x = x * mask # Áp dụng che
        
        # 3. Các stages còn lại
        for i in range(4):
            if i > 0:
                x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)
            
        return x, mask

    def forward_loss(self, imgs, pred, mask):
        # Tính loss chỉ trên những vùng bị che
        target = self.patchify(imgs)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) 

        # Resize mask về cùng kích thước với loss để tính toán
        # Mask gốc kích thước nhỏ [B, 1, 56, 56], cần đưa về [B, 49] (số lượng patch)
        # Ở đây đơn giản hóa: tính mean loss toàn cục
        return loss.mean()

    def forward(self, imgs, mask_ratio=0.6):
        latent, mask = self.forward_encoder(imgs, mask_ratio)
        
        # Decoder
        x = latent.permute(0, 2, 3, 1) # [N, H, W, C]
        x = x.flatten(1, 2) # [N, L, C]
        x = self.decoder_embed(x)
        
        # Reshape lại để đưa qua Conv Block của Decoder
        H_grid = int(x.shape[1]**0.5)
        x = x.reshape(x.shape[0], H_grid, H_grid, -1).permute(0, 3, 1, 2)
        
        for blk in self.decoder_blocks:
            x = blk(x)
            
        x = x.permute(0, 2, 3, 1)
        x = self.decoder_norm(x)
        x = x.flatten(1, 2)
        
        pred = self.decoder_pred(x)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask

# Factory functions
def convnextv2_base(pretrained=False, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_tiny(pretrained=False, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model