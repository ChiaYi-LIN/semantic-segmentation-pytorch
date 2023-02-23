from functools import partial
from typing import Optional, Callable, List, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class SSformerDecoder(nn.Module):
    """
    Implements SSformer Decoder Head.
    Args:
        num_classes (int): Number of classes for classification head. Default: 1000.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
    """

    def __init__(
        self,
        head: str,
        num_class: int = 150,
        use_softmax: bool = False,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        version: str = "v1",
    ):
        super(SSformerDecoder, self).__init__()
        self.use_softmax = use_softmax
        self.head = head
        self.version = version

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)        
        
        self.encode1_proj = nn.Sequential(
            norm_layer(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )
        self.encode2_proj = nn.Sequential(
            norm_layer(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU()
        )
        self.encode3_proj = nn.Sequential(
            norm_layer(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.GELU()
        )
        self.encode4_proj = nn.Sequential(
            norm_layer(embed_dim * 8),
            nn.Linear(embed_dim * 8, embed_dim),
            nn.GELU()
        )

        self.fusion = nn.Sequential(
            norm_layer(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim * 4),
            nn.GELU()
        )

        if head == "SSSR":
            self.last = SSformerSSHead(in_dim=embed_dim * 4, num_class=num_class, use_softmax=use_softmax)
        elif head == "SISR":
            self.last = SSformerSRHead(in_dim=embed_dim * 4, out_channel=3)
        else:
            raise Exception("Invalid decoder head")
        
    def forward(self, feat_out, segSize=None, return_feature_maps=False):
        assert(len(feat_out) == 4)
        encode1, encode2, encode3, encode4 = feat_out
        encode1, encode2, encode3, encode4 = encode1.permute(0, 2, 3, 1), encode2.permute(0, 2, 3, 1), encode3.permute(0, 2, 3, 1), encode4.permute(0, 2, 3, 1)
        
        decode1, decode2, decode3, decode4 = self.encode1_proj(encode1), self.encode2_proj(encode2), self.encode3_proj(encode3), self.encode4_proj(encode4)   

        feature_list = [decode1.permute(0, 3, 1, 2), decode2.permute(0, 3, 1, 2), decode3.permute(0, 3, 1, 2), decode4.permute(0, 3, 1, 2)]
        output_size = feature_list[0].size()[2:]
        fusion_list = [feature_list[0]]
        for i in range(1, len(feature_list)):
            fusion_list.append(nn.functional.interpolate(
                feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1) # N C H W
        
        fusion_out = self.fusion(fusion_out.permute(0, 2, 3, 1))
        # upsample_out = self.upsample(fusion_out)
        x = self.last(fusion_out.permute(0, 3, 1, 2), segSize)

        if return_feature_maps:
            return x, fusion_out
        else:
            return x


class SSformerSSHead(nn.Module):
    def __init__(self, in_dim, num_class, use_softmax) -> None:
        super().__init__()
        self.use_softmax = use_softmax

        self.conv_last = nn.Conv2d(in_dim, num_class, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x, segSize=None):
        """
        input: N, C, H, W
        output: N, C, H, W
        """
        x = self.conv_last(x)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)

        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
    

class SSformerSRHead(nn.Module):
    def __init__(self, in_dim, out_channel) -> None:
        super().__init__()
        
        self.conv_last = nn.Conv2d(in_dim, out_channel, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x: Tensor, segSize: tuple = None):
        """
        input: N, C, H, W
        output: N, C, H, W
        """
        x = self.conv_last(x)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)        

        return x    


class Upsample(nn.Module):
    def __init__(self, in_dim, out_dim, up_scale) -> None:
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_dim, (up_scale ** 2) * out_dim, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(up_scale),
            nn.GELU(),
        )

    def forward(self, x: Tensor):
        """
        input: N, H, W, C
        output: N, H, W, C
        """
        x = self.upsample(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        return x

