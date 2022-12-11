from functools import partial
from typing import Optional, Callable, List, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from torchvision.models.swin_transformer import StochasticDepth, MLP
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
BatchNorm2d = SynchronizedBatchNorm2d


__all__ = ['swin_t', 'swin_s', 'swin_b']


def swin_t(pretrained=False, **kwargs):
    """Constructs a swin_t model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    if pretrained:
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_t(weights=None)
    return model


def swin_s(pretrained=False, **kwargs):
    """Constructs a swin_s model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    if pretrained:
        model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_s(weights=None)
    return model


def swin_b(pretrained=False, **kwargs):
    """Constructs a swin_b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    
    if pretrained:
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    else:
        model = models.swin_b(weights=None)
    return model


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def transconv3x3_bn_relu(in_planes, out_planes, stride=2):
    "3x3 transpose convolution + BN + relu"
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3,
            stride=stride, padding=1, output_padding=1, bias=False),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class CustomTransforms:
    def __init__(self) -> None:
        self.inverse_normalize_transform = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std = [1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std = [1., 1., 1.]),
        ])

class SwinTransformerDecoder(nn.Module):
    """
    Implements Swin Transformer Decoder Head.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.0.
        num_classes (int): Number of classes for classification head. Default: 1000.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(
        self,
        head: str,
        num_class: int = 150,
        use_softmax: bool = False,
        patch_size: List[int] = [4, 4],
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 2, 2],
        num_heads: List[int] = [24, 12, 6, 3],
        window_size: List[int] = [7, 7],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        fpn_dim: int = 256,
    ):
        super(SwinTransformerDecoder, self).__init__()
        self.use_softmax = use_softmax

        if block is None:
            block = SASwinTransformerBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        stage_dims = (8 * embed_dim, 4 * embed_dim, 2 * embed_dim, embed_dim)
        layers: List[nn.Module] = []

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = stage_dims[i_stage]
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch splitting layer
            if i_stage < (len(depths) - 1):
                layers.append(TransformerUpsample(dim, norm_layer))

        self.body = nn.Sequential(*layers)
        self.norm4 = norm_layer(8 * embed_dim)
        self.norm3 = norm_layer(4 * embed_dim)
        self.norm2 = norm_layer(2 * embed_dim)
        self.norm1 = norm_layer(embed_dim)
        
        if head == "SSSR":
            self.last = SwinTransformerDecoderSSSRHead(in_dim=sum(stage_dims), num_class=num_class, use_softmax=use_softmax, hidden_dim=fpn_dim, norm_layer=norm_layer)
        elif head == "SISR":
            self.last = SwinTransformerDecoderSISRHead(in_dim=sum(stage_dims), out_channel=3, hidden_dim=fpn_dim, norm_layer=norm_layer)
        else:
            raise Exception("Invalid decoder head")
        
    def forward(self, feat_out, segSize=None):
        assert(len(feat_out) == 4)
        encode1, encode2, encode3, encode4 = feat_out
        encode1, encode2, encode3, encode4 = encode1.permute(0, 2, 3, 1), encode2.permute(0, 2, 3, 1), encode3.permute(0, 2, 3, 1), encode4.permute(0, 2, 3, 1)
        
        decode4 = self.body[0](encode4)   
        decode3 = self.body[2](self.body[1](decode4, target_size=(encode3.shape[1], encode3.shape[2])) + encode3)
        decode2 = self.body[4](self.body[3](decode3, target_size=(encode2.shape[1], encode2.shape[2])) + encode2)
        decode1 = self.body[6](self.body[5](decode2, target_size=(encode1.shape[1], encode1.shape[2])) + encode1)
        
        decode4 = self.norm4(decode4)
        decode3 = self.norm3(decode3)
        decode2 = self.norm2(decode2)
        decode1 = self.norm1(decode1)

        fpn_feature_list = [decode1.permute(0, 3, 1, 2), decode2.permute(0, 3, 1, 2), decode3.permute(0, 3, 1, 2), decode4.permute(0, 3, 1, 2)]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1) # N C H W
        
        x = self.last(fusion_out, segSize)

        return x


class SwinTransformerDecoderSSSRHead(nn.Module):
    def __init__(self, in_dim, num_class, use_softmax, hidden_dim, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        self.use_softmax = use_softmax

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.layers = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, num_class),
        )
    
    def forward(self, x, segSize=None):
        """
        input: N, C, H, W
        output: N, C, H, W
        """
        x = self.layers(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)

        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
    

class SwinTransformerDecoderSISRHead(nn.Module):
    def __init__(self, in_dim, out_channel, hidden_dim, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)
        self.project = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                norm_layer(hidden_dim),
        )
        self.layers = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_channel),
        )
        self.inverse_normalize_transform = CustomTransforms().inverse_normalize_transform
    
    def forward(self, x, segSize=None):
        """
        input: N, C, H, W
        output: N, C, H, W
        """
        x = self.project(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)

        x = self.layers(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.inverse_normalize_transform(x)

        return x    


class TransformerUpsample(nn.Module):
    """Bilinear Upsample Layer for Transformer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim, dim // 2, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: Tensor, target_size: tuple = None):
        """
        Args:
            x (Tensor): input tensor with expected layout of [N, H, W, C]
            target_size (tuple): target upsample size (if None then (2*H, 2*W))
        Returns:
            Tensor with layout of [N, 2*H, 2*W, C/2]
        """
        x = x.permute(0, 3, 1, 2)
        N, C, H, W = x.shape
        if target_size is None:
            x = nn.functional.interpolate(x, size=(H * 2, W * 2), mode='bilinear', align_corners=False)
        else:
            x = nn.functional.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = self.reduction(x)  # N 2*H 2*W C/2
        return x

class PatchSpliting(nn.Module):
    """Patch Spliting Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    """

    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expansion = nn.Linear(dim // 4, dim // 2, bias=False)
        self.norm = norm_layer(dim // 4)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [N, H, W, C]
        Returns:
            Tensor with layout of [N, 2*H, 2*W, C/2]
        """
        N, H, W, C = x.shape
        x0 = x[..., :C//4*1]
        x1 = x[..., C//4*1:C//4*2]
        x2 = x[..., C//4*2:C//4*3]
        x3 = x[..., C//4*3:]

        y = x0.repeat(1, 2, 2, 1) # N 2*H 2*W C/4
        y[..., 0::2, 0::2, :] = x0
        y[..., 1::2, 0::2, :] = x1
        y[..., 0::2, 1::2, :] = x2
        y[..., 1::2, 1::2, :] = x3

        y = self.norm(y)
        y = self.expansion(y)  # N 2*H 2*W C/2
        return y

    
def sa_shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    shared_attn: Tensor = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5

    if shared_attn is not None:
        attn = shared_attn
    else:
        attn = q.matmul(k.transpose(-2, -1))
        # add relative position bias
        attn = attn + relative_position_bias

        if sum(shift_size) > 0:
            # generate attention mask
            attn_mask = x.new_zeros((pad_H, pad_W))
            h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
            w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
            count = 0
            for h in h_slices:
                for w in w_slices:
                    attn_mask[h[0] : h[1], w[0] : w[1]] = count
                    count += 1
            attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
            attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
            attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
            attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, num_heads, x.size(1), x.size(1))

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x, attn


class SAShiftedWindowAttention(nn.Module):
    """
    See :func:`sa_shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)


    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        x, attn_score = sa_shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            shared_attn=None
        )


        return x
    

class SASwinTransformerBlock(nn.Module):
    """
    Shared Attention Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ModifiedShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = SAShiftedWindowAttention,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x

    
if __name__ == "__main__":
    model = swin_t(pretrained=True)
    print(model)