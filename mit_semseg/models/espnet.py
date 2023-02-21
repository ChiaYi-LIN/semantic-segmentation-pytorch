"""
Code references:
https://github.com/Xilinx/DSRL
https://github.com/sacmehta/EdgeNets
"""

from functools import partial
from typing import Optional, Callable, List, Any
import math
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from mit_semseg.models.utils import load_url


__all__ = ['espnetv2']


model_urls = {
    'espnetv2': 'https://github.com/sacmehta/ESPNetv2/raw/master/imagenet/pretrained_weights/espnetv2_s_2.0.pth',
}


def espnetv2(pretrained=False, **kwargs):
    """Constructs a espnetv2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ESPNetv2(s=2.0, num_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['espnetv2']), strict=False)

    return model


def Deconv_BN_ACT(in_plane, out_plane):
    conv_trans = nn.ConvTranspose2d(in_plane, out_plane ,kernel_size=3, stride=2,padding=1, output_padding=1, bias=False)
    norm =  nn.BatchNorm2d(out_plane)
    act = nn.PReLU(out_plane)
    return nn.Sequential(conv_trans, norm, act)


def activation_fn(features, name='prelu', inplace=True):
    '''
    :param features: # of features (only for PReLU)
    :param name: activation name (prelu, relu, selu)
    :param inplace: Inplace operation or not
    :return:
    '''
    if name == 'relu':
        return nn.ReLU(inplace=True)
    elif name == 'selu':
        return nn.SELU(inplace=inplace)
    elif name == 'prelu':
        return nn.PReLU(features)
    else:
        NotImplementedError('Not implemented yet')
        exit()


class CBR(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and activation function
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        :param act_name: Name of the activation function
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cbr = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=dilation),
            nn.BatchNorm2d(nOut),
            activation_fn(features=nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cbr(x)
    

class CB(nn.Module):
    '''
    This class implements convolution layer followed by batch normalization
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        :param groups: # of groups for group-wise convolution
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)*dilation
        self.cb = nn.Sequential(
            nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups, dilation=1),
            nn.BatchNorm2d(nOut),
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.cb(x)


class BR(nn.Module):
    '''
    This class implements batch normalization and  activation function
    '''
    def __init__(self, nOut, act_name='prelu'):
        '''
        :param nIn: number of input channels
        :param act_name: Name of the activation function
        '''
        super().__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(nOut),
            activation_fn(nOut, name=act_name)
        )

    def forward(self, x):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        return self.br(x)


class Shuffle(nn.Module):
    '''
    This class implements Channel Shuffling
    '''
    def __init__(self, groups):
        '''
        :param groups: # of groups for shuffling
        '''
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = x.view(batchsize, self.groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        return x
    

class EfficientPWC(nn.Module):
    def __init__(self, nin, nout):
        super(EfficientPWC, self).__init__()
        self.wt_layer = nn.Sequential(
                        nn.AdaptiveAvgPool2d(output_size=1),
                        nn.Conv2d(nin, nout, kernel_size=1, stride=1, padding=0, groups=1, bias=False),
                        nn.Sigmoid()
                    )

        self.groups = math.gcd(nin, nout)
        self.expansion_layer = CBR(nin, nout, kSize=3, stride=1, groups=self.groups)

        self.out_size = nout
        self.in_size = nin

    def forward(self, x):
        wts = self.wt_layer(x)
        x = self.expansion_layer(x)
        x = x * wts
        return x

    def __repr__(self):
        s = '{name}(in_channels={in_size}, out_channels={out_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class EfficientPyrPool(nn.Module):
    """Efficient Pyramid Pooling Module"""

    def __init__(self, in_planes, proj_planes, out_planes, scales=[2.0, 1.5, 1.0, 0.5, 0.1], last_layer_br=True):
        super(EfficientPyrPool, self).__init__()
        self.stages = nn.ModuleList()
        scales.sort(reverse=True)

        self.projection_layer = CBR(in_planes, proj_planes, 1, 1)
        for _ in enumerate(scales):
            self.stages.append(nn.Conv2d(proj_planes, proj_planes, kernel_size=3, stride=1, padding=1, bias=False, groups=proj_planes))

        self.merge_layer = nn.Sequential(
            # perform one big batch normalization instead of p small ones
            BR(proj_planes * len(scales)),
            Shuffle(groups=len(scales)),
            CBR(proj_planes * len(scales), proj_planes, 3, 1, groups=proj_planes),
            nn.Conv2d(proj_planes, out_planes, kernel_size=1, stride=1, bias=not last_layer_br),
        )
        if last_layer_br:
            self.br = BR(out_planes)
        self.last_layer_br = last_layer_br
        self.scales = scales

    def forward(self, x):
        hs = []
        x = self.projection_layer(x)
        height, width = x.size()[2:]
        for i, stage in enumerate(self.stages):
            h_s = int(math.ceil(height * self.scales[i]))
            w_s = int(math.ceil(width * self.scales[i]))
            h_s = h_s if h_s > 5 else 5
            w_s = w_s if w_s > 5 else 5
            if self.scales[i] < 1.0:
                h = F.adaptive_avg_pool2d(x, output_size=(h_s, w_s))
                h = stage(h)
                h = F.interpolate(h, (height, width), mode='bilinear', align_corners=True)
            elif self.scales[i] > 1.0:
                h = F.interpolate(x, (h_s, w_s), mode='bilinear', align_corners=True)
                h = stage(h)
                h = F.adaptive_avg_pool2d(h, output_size=(height, width))
            else:
                h = stage(x)
            hs.append(h)

        out = torch.cat(hs, dim=1)
        out = self.merge_layer(out)
        if self.last_layer_br:
            return self.br(out)
        return out
    

class CDilated(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output
    

class EESP(nn.Module):
    '''
    This class defines the EESP block, which is based on the following principle
        REDUCE ---> SPLIT ---> TRANSFORM --> MERGE
    '''

    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'): #down_method --> ['avg' or 'esp']
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param stride: factor by which we should skip (useful for down-sampling). If 2, then down-samples the feature map by 2
        :param k: # of parallel branches
        :param r_lim: A maximum value of receptive field allowed for EESP block
        :param down_method: Downsample or not (equivalent to say stride is 2 or not)
        '''
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        # (For convenience) Mapping between dilation rate and receptive field for a 3x3 kernel
        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            # After reaching the receptive field limit, fall back to the base kernel size of 3 with a dilation rate of 1
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        # sort (in ascending order) these kernel sizes based on their receptive field
        # This enables us to ignore the kernels (3x3 in our case) with the same effective receptive field in hierarchical
        # feature fusion because kernels with 3x3 receptive fields does not have gridding artifact.
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        # Performing a group convolution with K groups is the same as performing K point-wise convolutions
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.PReLU(nOut)
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''

        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        # compute the output for each branch and hierarchically fuse them
        # i.e. Split --> Transform --> HFF
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            # HFF
            out_k = out_k + output[k - 1]
            output.append(out_k)
        # Merge
        expanded = self.conv_1x1_exp( # learn linear combinations using group point-wise convolutions
            self.br_after_cat( # apply batch normalization followed by activation function (PRelu in this case)
                torch.cat(output, 1) # concatenate the output of different branches
            )
        )
        del output
        # if down-sampling, then return the concatenated vector
        # because Downsampling function will combine it with avg. pooled feature map and then threshold it
        if self.stride == 2 and self.downAvg:
            return expanded

        # if dimensions of input and concatenated vector are the same, add them (RESIDUAL LINK)
        if expanded.size() == input.size():
            expanded = expanded + input

        # Threshold the feature map using activation function (PReLU in this case)
        return self.module_act(expanded)


class DownSampler(nn.Module):
    '''
    Down-sampling fucntion that has three parallel branches: (1) avg pooling,
    (2) EESP block with stride of 2 and (3) efficient long-range connection with the input.
    The output feature maps of branches from (1) and (2) are concatenated and then additively fused with (3) to produce
    the final output.
    '''

    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        '''
            :param nin: number of input channels
            :param nout: number of output channels
            :param k: # of parallel branches
            :param r_lim: A maximum value of receptive field allowed for EESP block
            :param reinf: Use long range shortcut connection with the input or not.
        '''
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        config_inp_reinf = 3
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(config_inp_reinf, config_inp_reinf, 3, 1),
                CB(config_inp_reinf, nout, 1, 1)
            )
        self.act =  nn.PReLU(nout)

    def forward(self, input, input2=None):
        '''
        :param input: input feature map
        :return: feature map down-sampled by a factor of 2
        '''
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)

        if input2 is not None:
            #assuming the input is a square image
            # Shortcut connection with the input image
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output)
    

class ESPNetv2(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the ImageNet classification
    '''

    def __init__(self, s=2.0, num_class=20, channels=3, input_reinforcement=True):
        '''
        :param num_class: number of classes in the dataset. Default is 1000 for the ImageNet dataset
        :param s: factor that scales the number of output feature maps
        '''
        super().__init__()

        # ====================
        # Network configuraiton
        # ====================
        try:
            num_class = num_class
        except:
            # if not specified, default to 1000 for imageNet
            num_class = 1000  # 1000 for imagenet

        try:
            channels_in = channels
        except:
            # if not specified, default to RGB (3)
            channels_in = 3

        sc_ch_dict = {
            0.5: [16, 32, 64, 128, 256, 1024],
            1.0: [32, 64, 128, 256, 512, 1024],
            1.25: [32, 80, 160, 320, 640, 1024],
            1.5: [32, 96, 192, 384, 768, 1024],
            2.0: [32, 128, 256, 512, 1024, 1280]
        }
        rep_layers = [0, 3, 7, 3]

        # limits for the receptive field at each spatial level
        recept_limit = [13, 11, 9, 7, 5]
        branches = 4

        if not s in sc_ch_dict.keys():
            print('Model at scale s={} is not suppoerted yet'.format(s))
            exit(-1)

        out_channel_map = sc_ch_dict[s]
        reps_at_each_level = rep_layers

        recept_limit = recept_limit  # receptive field at each spatial level
        K = [branches]*len(recept_limit) # No. of parallel branches at different level

        # True for the shortcut connection with input
        self.input_reinforcement = input_reinforcement

        assert len(K) == len(recept_limit), 'Length of branching factor array and receptive field array should be the same.'

        self.level1 = CBR(channels_in, out_channel_map[0], 3, 2)  # 112 L1

        self.level2_0 = DownSampler(out_channel_map[0], out_channel_map[1], k=K[0], r_lim=recept_limit[0], reinf=self.input_reinforcement)  # out = 56

        self.level3_0 = DownSampler(out_channel_map[1], out_channel_map[2], k=K[1], r_lim=recept_limit[1], reinf=self.input_reinforcement) # out = 28
        self.level3 = nn.ModuleList()
        for i in range(reps_at_each_level[1]):
            self.level3.append(EESP(out_channel_map[2], out_channel_map[2], stride=1, k=K[2], r_lim=recept_limit[2]))

        self.level4_0 = DownSampler(out_channel_map[2], out_channel_map[3], k=K[2], r_lim=recept_limit[2], reinf=self.input_reinforcement) #out = 14
        self.level4 = nn.ModuleList()
        for i in range(reps_at_each_level[2]):
            self.level4.append(EESP(out_channel_map[3], out_channel_map[3], stride=1, k=K[3], r_lim=recept_limit[3]))

        self.level5_0 = DownSampler(out_channel_map[3], out_channel_map[4], k=K[3], r_lim=recept_limit[3]) #7
        self.level5 = nn.ModuleList()
        for i in range(reps_at_each_level[3]):
            self.level5.append(EESP(out_channel_map[4], out_channel_map[4], stride=1, k=K[4], r_lim=recept_limit[4]))

        # expand the feature maps using depth-wise convolution followed by group point-wise convolution
        self.level5.append(CBR(out_channel_map[4], out_channel_map[4], 3, 1, groups=out_channel_map[4]))
        self.level5.append(CBR(out_channel_map[4], out_channel_map[5], 1, 1, groups=K[4]))

        self.classifier = nn.Linear(out_channel_map[5], num_class)
        self.config = out_channel_map
        self.init_params()

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, p=0.2):
        '''
        :param input: Receives the input RGB image
        :return: a C-dimensional vector, C=# of classes
        '''
        out_l1 = self.level1(input)  # 112
        if not self.input_reinforcement:
            del input
            input = None

        out_l2 = self.level2_0(out_l1, input)  # 56

        out_l3_0 = self.level3_0(out_l2, input)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)

        out_l4_0 = self.level4_0(out_l3, input)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)

        out_l5_0 = self.level5_0(out_l4)  # down-sample
        for i, layer in enumerate(self.level5):
            if i == 0:
                out_l5 = layer(out_l5_0)
            else:
                out_l5 = layer(out_l5)

        output_g = F.adaptive_avg_pool2d(out_l5, output_size=1)
        output_g = F.dropout(output_g, p=p, training=self.training)
        output_1x1 = output_g.view(output_g.size(0), -1)

        return self.classifier(output_1x1)


class ESPNetv2Decoder(nn.Module):
    '''
    This class defines the ESPNetv2 architecture for the Semantic Segmenation
    '''

    def __init__(
        self,
        head: str,
        num_class: int = 19, 
        use_softmax: bool = False, 
        s: float = 2.0, 
        dataset: str = 'city'
    ):
        super().__init__()
        self.use_softmax = use_softmax

        sc_ch_dict = {
            0.5: [16, 32, 64, 128, 256, 1024],
            1.0: [32, 64, 128, 256, 512, 1024],
            1.25: [32, 80, 160, 320, 640, 1024],
            1.5: [32, 96, 192, 384, 768, 1024],
            2.0: [32, 128, 256, 512, 1024, 1280]
        }
        config = sc_ch_dict[s]

        #=============================================================
        #                   SEGMENTATION BASE NETWORK
        #=============================================================
        #
        #  same as Line36-66 in ESPNetv2      
        #   
        dec_feat_dict={
            'pascal': 16,
            'city': 16,
            'coco': 32
        }
        base_dec_planes = dec_feat_dict[dataset]
        dec_planes = [4*base_dec_planes, 3*base_dec_planes, 2*base_dec_planes, num_class]
        pyr_plane_proj = min(num_class //2, base_dec_planes)

        self.bu_dec_l1 = EfficientPyrPool(in_planes=config[3], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[0])
        self.bu_dec_l2 = EfficientPyrPool(in_planes=dec_planes[0], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[1])
        self.bu_dec_l3 = EfficientPyrPool(in_planes=dec_planes[1], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[2])
        self.bu_dec_l4 = EfficientPyrPool(in_planes=dec_planes[2], proj_planes=pyr_plane_proj,
                                          out_planes=dec_planes[3], last_layer_br=False)

        self.merge_enc_dec_l2 = EfficientPWC(config[2], dec_planes[0])
        self.merge_enc_dec_l3 = EfficientPWC(config[1], dec_planes[1])
        self.merge_enc_dec_l4 = EfficientPWC(config[0], dec_planes[2])

        self.bu_br_l2 = nn.Sequential(nn.BatchNorm2d(dec_planes[0]),
                                      nn.PReLU(dec_planes[0])
                                      )
        self.bu_br_l3 = nn.Sequential(nn.BatchNorm2d(dec_planes[1]),
                                      nn.PReLU(dec_planes[1])
                                      )
        self.bu_br_l4 = nn.Sequential(nn.BatchNorm2d(dec_planes[2]),
                                      nn.PReLU(dec_planes[2])
                                      )
        
        # Upsample        
        self.bu_dec_l5 = Deconv_BN_ACT(dec_planes[3],dec_planes[3])
        self.bu_dec_l6 = Deconv_BN_ACT(dec_planes[3],dec_planes[3])
        
        if head == "SSSR":
            self.last = ESPNetv2DecoderSS(dec_planes[3], dec_planes[3], use_softmax=use_softmax)
        elif head == "SISR":
            self.last = ESPNetv2DecoderSR(dec_planes[3], 3)
        else:
            raise Exception("Invalid decoder head")
        
        self.init_params()

    def upsample(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

    def init_params(self):
        '''
        Function to initialze the parameters
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, conv_out, segSize=None, return_feature_maps=False):
        assert(len(conv_out) == 4)
        encode1, encode2, encode3, encode4 = conv_out

        # bottom-up decoding
        ###### this model outputs: bu_out0
        bu_out = self.bu_dec_l1(encode4)
        # print(f"bu_out0.shape = {bu_out.shape}")

        # Decoding block
        ###### this model outputs: bu_out1
        bu_out = self.upsample(bu_out)
        enc_out_l3_proj = self.merge_enc_dec_l2(encode3)
        bu_out = enc_out_l3_proj + bu_out
        bu_out = self.bu_br_l2(bu_out)
        bu_out = self.bu_dec_l2(bu_out)
        # print(f"bu_out1.shape = {bu_out.shape}")

        # Decoding block
        ###### this model outputs: bu_out2
        bu_out = self.upsample(bu_out)
        enc_out_l2_proj = self.merge_enc_dec_l3(encode2)
        bu_out = enc_out_l2_proj + bu_out
        bu_out = self.bu_br_l3(bu_out)
        bu_out = self.bu_dec_l3(bu_out)
        # print(f"bu_out2.shape = {bu_out.shape}")
        
        # Decoding block
        ###### this model outputs: bu_out
        bu_out = self.upsample(bu_out)
        enc_out_l1_proj = self.merge_enc_dec_l4(encode1)
        bu_out = enc_out_l1_proj + bu_out
        bu_out = self.bu_br_l4(bu_out)
        bu_out  = self.bu_dec_l4(bu_out)
        # print(f"bu_out.shape = {bu_out.shape}")
        # return F.interpolate(bu_out, size=(x_size[0]*2, x_size[1]*2), mode='bilinear', align_corners=True)

        # sssr block
        up_out1 = self.bu_dec_l5(bu_out)
        up_out2 = self.bu_dec_l6(up_out1)
        # print(f"up_out1.shape = {up_out1.shape}")
        # print(f"up_out2.shape = {up_out2.shape}")

        x = self.last(up_out2, segSize=segSize) 

        if return_feature_maps:
            return x, up_out2
        return  x 

class ESPNetv2DecoderSS(nn.Module):
    def __init__(self, in_dim, num_class, use_softmax) -> None:
        super().__init__()
        self.use_softmax = use_softmax

        self.conv_last = nn.Conv2d(in_dim, num_class, kernel_size=3, stride=1, padding=1)
    
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
    

class ESPNetv2DecoderSR(nn.Module):
    def __init__(self, in_dim, out_channel=3) -> None:
        super().__init__()
        
        self.conv_last = nn.Conv2d(in_dim, out_channel, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x: Tensor, segSize: tuple = None):
        """
        input: N, C, H, W
        output: N, C, H, W
        """
        x = self.conv_last(x)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)        

        return x    
    

if __name__ == "__main__":
    model = espnetv2(pretrained=True)
    print(model)