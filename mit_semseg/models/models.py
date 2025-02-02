import numpy as np
import torch
import torch.nn as nn
from . import resnet, resnext, mobilenet, hrnet, swintransformer, espnet, ssformer
from mit_semseg.lib.nn import SynchronizedBatchNorm2d
from mit_semseg.lib.utils.calculate_psnr_ssim import Postprocess
BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()
        self.postprocess = Postprocess()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_encoder, net_decoder_ss, net_decoder_sr, crit_ss=None, crit_sr=None, crit_aff=None, options=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_encoder
        self.decoder_ss = net_decoder_ss
        self.decoder_sr = net_decoder_sr
        self.crit_ss = crit_ss
        self.crit_sr = crit_sr
        self.crit_aff = crit_aff
        if options is not None:
            self.aff_loss = options.get("aff_loss", None)
            self.w_1 = options.get("w_1", None)
            self.w_2 = options.get("w_2", None)
            self.w_3 = options.get("w_3", None)
            self.deep_sup_scale = options.get("deep_sup_scale", None)

    def forward(self, feed_dict, *, segSize=None):
        img_data = feed_dict['img_data']
        input_img = nn.functional.interpolate(img_data, size=(img_data.shape[2] // 2, img_data.shape[3] // 2), mode='bicubic', align_corners=False)
        # training
        if segSize is None:
            # Loss Tracker
            loss = None
            loss_dict = {
                "acc": None,
                "ss": None,
                "sr": None,
                "aff": None,
                "psnr" : None,
            }

            # Shared Encoder
            encode_feats = self.encoder(input_img, return_feature_maps=True)

            # SS Path
            if self.decoder_ss is not None:
                seg_label = feed_dict['seg_label']
                if self.deep_sup_scale is not None: # use deep supervision technique
                    (pred, pred_deepsup) = self.decoder_ss(encode_feats)
                else:
                    pred_ss, decode_feats_ss = self.decoder_ss(encode_feats, segSize=(seg_label.shape[1], seg_label.shape[2]), return_feature_maps=True)
                
                if self.deep_sup_scale is not None:
                    loss_ss = self.crit_ss(pred, seg_label) + self.deep_sup_scale * self.crit_ss(pred_deepsup, seg_label)
                else:
                    loss_ss = self.crit_ss(pred_ss, seg_label)
                
                loss_dict["ss"] = loss_ss
                loss_dict["acc"] = self.pixel_acc(pred, seg_label)

                if loss is None:
                    loss = self.w_1 * loss_ss
                else:
                    loss += self.w_1 * loss_ss

            # SR Path
            if self.decoder_sr is not None:
                pred_sr, decode_feats_sr = self.decoder_sr(encode_feats, segSize=(img_data.shape[2], img_data.shape[3]), return_feature_maps=True)
                loss_sr = self.crit_sisr(self.postprocess.inverse2rgb(pred_sr), self.postprocess.inverse2rgb(img_data))
                loss_dict["sr"] = loss_sr

                if loss is None:
                    loss = self.w_2 * loss_sr
                else:
                    loss += self.w_2 * loss_sr

            # Affinity
            if self.decoder_ss is not None and self.decoder_sr is not None and self.crit_aff is not None:
                if self.aff_loss == "aa":
                    attns_ss = []
                    attns_sr = []
                    stage_indices = [0, 2, 4, 6]
                    block_indices = [0, 1]
                    for stage_index in stage_indices:
                        for block_index in block_indices:
                            attn_ss = self.decoder_ss.body[stage_index][block_index].attn.attn_score
                            attn_sr = self.decoder_sr.body[stage_index][block_index].attn.attn_score
                            attns_ss.append(attn_ss.view(-1, attn_ss.shape[2], attn_ss.shape[3]))
                            attns_sr.append(attn_sr.view(-1, attn_sr.shape[2], attn_sr.shape[3]))
                    loss_aff = self.crit_aff(torch.cat(attns_ss, dim=0), torch.cat(attns_sr, dim=0))
                    loss_dict["aff"] = loss_aff

                    if loss is None:
                        loss = self.w_3 * loss_aff
                    else:
                        loss += self.w_3 * loss_aff

            return loss, loss_dict
        # inference
        else:
            output_dict = {
                "segment": None,
                "reconstruct": None,
            }
            encode_feats = self.encoder(input_img, return_feature_maps=True)
            if self.decoder_ss is not None:
                output_dict["segment"] = self.decoder_ss(encode_feats, segSize=segSize)
            if self.decoder_sr is not None:
                output_dict["reconstruct"] = self.decoder_sr(encode_feats, segSize=segSize)
            return output_dict


class SegmentationModuleCity(SegmentationModule):
    def __init__(self, net_encoder, net_decoder_ss, net_decoder_sr, crit_ss=None, crit_sr=None, crit_aff=None, options=None):
        super(SegmentationModuleCity, self).__init__(net_encoder, net_decoder_ss, net_decoder_sr, crit_ss, crit_sr, crit_aff, options)

    def forward(self, feed_dict, *, segSize=None):
        img_data = feed_dict['img_data']
        
        # training
        if segSize is None:
            # Loss Tracker
            loss = None
            loss_dict = {
                "acc": None,
                "ss": None,
                "sr": None,
                "aff": None,
                "psnr" : None,
            }

            # Shared Encoder
            encode_feats = self.encoder(img_data, return_feature_maps=True)

            # SS Path
            if self.decoder_ss is not None:
                seg_label = feed_dict['seg_label']
                pred_ss, decode_feats_ss = self.decoder_ss(encode_feats, segSize=(seg_label.shape[1], seg_label.shape[2]), return_feature_maps=True)
                loss_ss = self.crit_ss(pred_ss, seg_label)
                
                loss_dict["ss"] = loss_ss
                loss_dict["acc"] = self.pixel_acc(pred_ss, seg_label)

                if loss is None:
                    loss = self.w_1 * loss_ss
                else:
                    loss += self.w_1 * loss_ss

            # SR Path
            if self.decoder_sr is not None:
                img_ori = feed_dict['img_ori']
                pred_sr, decode_feats_sr = self.decoder_sr(encode_feats, segSize=(img_ori.shape[2], img_ori.shape[3]), return_feature_maps=True)
                loss_sr = self.crit_sr(self.postprocess.inverse2rgb(pred_sr), self.postprocess.inverse2rgb(img_ori))
                loss_dict["sr"] = loss_sr

                if loss is None:
                    loss = self.w_2 * loss_sr
                else:
                    loss += self.w_2 * loss_sr

            # Affinity
            if self.decoder_ss is not None and self.decoder_sr is not None and self.crit_aff is not None:
                if self.aff_loss == "aa":
                    attns_ss = []
                    attns_sr = []
                    stage_indices = [0, 2, 4, 6]
                    block_indices = [0, 1]
                    for stage_index in stage_indices:
                        for block_index in block_indices:
                            attn_ss = self.decoder_ss.body[stage_index][block_index].attn.attn_score
                            attn_sr = self.decoder_sr.body[stage_index][block_index].attn.attn_score
                            attns_ss.append(attn_ss.view(-1, attn_ss.shape[2], attn_ss.shape[3]))
                            attns_sr.append(attn_sr.view(-1, attn_sr.shape[2], attn_sr.shape[3]))
                    loss_aff = self.crit_aff(torch.cat(attns_ss, dim=0), torch.cat(attns_sr, dim=0))
                    loss_dict["aff"] = loss_aff

                elif self.aff_loss == "fa":
                    decode_feats_ss = torch.nn.functional.normalize(decode_feats_ss, p=2, dim=-1).view(decode_feats_ss.shape[0], -1, decode_feats_ss.shape[3])
                    decode_feats_sr = torch.nn.functional.normalize(decode_feats_sr, p=2, dim=-1).view(decode_feats_sr.shape[0], -1, decode_feats_sr.shape[3])
                    sim_matrix_ss = torch.bmm(decode_feats_ss, decode_feats_ss.permute(0, 2, 1))
                    sim_matrix_sr = torch.bmm(decode_feats_sr, decode_feats_sr.permute(0, 2, 1))
                    loss_aff = self.crit_aff(sim_matrix_ss, sim_matrix_sr)
                    loss_dict["aff"] = loss_aff

                if loss is None:
                    loss = self.w_3 * loss_aff
                else:
                    loss += self.w_3 * loss_aff
    
            # batch_size = pred_sisr.shape[0]
            # psnr = 0.0
            # for i in range(batch_size):
            #     rgb_pred = self.postprocess.inverse2pil(pred_sisr[i])
            #     rgb_gt = self.postprocess.inverse2pil(img_ori[i])
            #     psnr += self.postprocess.psnr(np.array(rgb_pred), np.array(rgb_gt))
            # loss_dict["psnr"] = torch.tensor(psnr / batch_size).to(img_ori.device)

            return loss, loss_dict

        # inference
        else:
            output_dict = {
                "segment": None,
                "reconstruct": None,
            }
            encode_feats = self.encoder(img_data, return_feature_maps=True)
            if self.decoder_ss is not None:
                output_dict["segment"] = self.decoder_ss(encode_feats, segSize=segSize)
            if self.decoder_sr is not None:
                output_dict["reconstruct"] = self.decoder_sr(encode_feats, segSize=segSize)
            return output_dict


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        elif arch == "espnetv2":
            orig_espnet = espnet.__dict__['espnetv2'](pretrained=pretrained)
            net_encoder = ESPNetv2(orig_espnet)
        elif arch == 'swin_t':
            orig_swin = swintransformer.__dict__['swin_t'](pretrained=pretrained)
            net_encoder = SwinTransformer(orig_swin)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder_ss(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        elif arch == "espnetv2":
            net_decoder = espnet.ESPNetv2Decoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax)
        elif arch == 'swin_t_v1':
            net_decoder = swintransformer.SwinTransformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax,
                version="v1")
        elif arch == 'swin_t_v2':
            net_decoder = swintransformer.SwinTransformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax,
                version="v2")
        elif arch == 'swin_t_v3':
            net_decoder = swintransformer.SwinTransformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax,
                version="v3")
        elif arch == 'swin_t_v4':
            net_decoder = swintransformer.SwinTransformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax,
                version="v4")
        elif arch == 'swin_t_v5':
            net_decoder = swintransformer.SwinTransformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax,
                version="v5")
        elif arch == 'ssformer':
            net_decoder = ssformer.SSformerDecoder(
                head="SSSR",
                num_class=num_class,
                use_softmax=use_softmax)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder
    
    
    @staticmethod
    def build_decoder_sr(arch='swin_t', weights=''):
        arch = arch.lower()
        if arch == "espnetv2":
            net_decoder_sisr = espnet.ESPNetv2Decoder(
                head="SISR")
        elif arch == 'swin_t_v1':
            net_decoder_sisr = swintransformer.SwinTransformerDecoder(
                head="SISR",
                version="v1")
        elif arch == 'swin_t_v2':
            net_decoder_sisr = swintransformer.SwinTransformerDecoder(
                head="SISR",
                version="v2")
        elif arch == 'swin_t_v3':
            net_decoder_sisr = swintransformer.SwinTransformerDecoder(
                head="SISR",
                version="v3")
        elif arch == 'swin_t_v4':
            net_decoder_sisr = swintransformer.SwinTransformerDecoder(
                head="SISR",
                version="v4")
        elif arch == 'swin_t_v5':
            net_decoder_sisr = swintransformer.SwinTransformerDecoder(
                head="SISR",
                version="v5")
        elif arch == 'ssformer':
            net_decoder_sisr = ssformer.SSformerDecoder(
                head="SISR")
        else:
            raise Exception('Architecture undefined!')
        
        net_decoder_sisr.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder_sisr')
            net_decoder_sisr.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder_sisr


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


class ESPNetv2(nn.Module):
    def __init__(self, orig_net):
        super(ESPNetv2, self).__init__()

        self.input_reinforcement = orig_net.input_reinforcement
        self.level1 = orig_net.level1
        self.level2_0 = orig_net.level2_0
        self.level3_0 = orig_net.level3_0
        self.level3 = orig_net.level3
        self.level4_0 = orig_net.level4_0
        self.level4 = orig_net.level4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        out_l1 = self.level1(x)  # 112
        if not self.input_reinforcement:
            del x
            x = None
        conv_out.append(out_l1)

        out_l2 = self.level2_0(out_l1, x)  # 56
        conv_out.append(out_l2)

        out_l3_0 = self.level3_0(out_l2, x)  # down-sample
        for i, layer in enumerate(self.level3):
            if i == 0:
                out_l3 = layer(out_l3_0)
            else:
                out_l3 = layer(out_l3)
        conv_out.append(out_l3)

        out_l4_0 = self.level4_0(out_l3, x)  # down-sample
        for i, layer in enumerate(self.level4):
            if i == 0:
                out_l4 = layer(out_l4_0)
            else:
                out_l4 = layer(out_l4)
        conv_out.append(out_l4)

        if return_feature_maps:
            return conv_out
        return [out_l4]

        
class SwinTransformer(nn.Module):
    def __init__(self, orig_swin):
        super(SwinTransformer, self).__init__()

        # take pretrained swin transformer, except AvgPool and FC
        # stage 1
        self.embed = orig_swin.features[0]
        self.layer1 = orig_swin.features[1]
        # stage 2
        self.merge1 = orig_swin.features[2]
        self.layer2 = orig_swin.features[3]
        # stage 3
        self.merge2 = orig_swin.features[4]
        self.layer3 = orig_swin.features[5]
        # stage 4
        self.merge3 = orig_swin.features[6]
        self.layer4 = orig_swin.features[7]
        self.norm = orig_swin.norm

    def forward(self, x, return_feature_maps=False):
        feat_out = []

        x = self.layer1(self.embed(x)); feat_out.append(x.permute(0, 3, 1, 2));
        x = self.layer2(self.merge1(x)); feat_out.append(x.permute(0, 3, 1, 2));
        x = self.layer3(self.merge2(x)); feat_out.append(x.permute(0, 3, 1, 2));
        x = self.layer4(self.merge3(x)); feat_out.append(x.permute(0, 3, 1, 2));

        if return_feature_maps:
            return feat_out
        return [x.permute(0, 3, 1, 2)]        


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        if segSize is not None:
            ppm_out = nn.functional.interpolate(
                ppm_out, size=segSize, mode='bilinear', align_corners=False)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        fpn_inplanes=(fc_dim // 8, fc_dim // 4, fc_dim // 2, fc_dim)
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)
        if x.shape[2] != segSize[0] or x.shape[3] != segSize[1]:
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)

        if self.use_softmax:  # is True during inference
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x
