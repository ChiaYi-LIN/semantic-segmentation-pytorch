# System libs
import os
import glob
import time
# import math
import random
import argparse
from packaging.version import Version
# Numerical libs
import torch
import torch.nn as nn
import torchvision
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, TrainDatasetSquareCrop
from mit_semseg.dataset_city import TrainDatasetCity
from mit_semseg.models import ModelBuilder, SegmentationModule, SegmentationModuleCity
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from mit_semseg.models.swintransformer import SAShiftedWindowAttention

# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_ss_loss = AverageMeter()
    ave_sr_loss = AverageMeter()
    ave_aff_loss = AverageMeter()
    ave_psnr = AverageMeter()

    segmentation_module.train(not cfg.TRAIN.fix_bn)

    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, loss_dict = segmentation_module(batch_data)
        loss = loss.mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            if optimizer is not None:
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        if loss_dict["acc"] is not None:
            ave_acc.update(loss_dict["acc"].mean().data.item()*100)
        if loss_dict["ss"] is not None:
            ave_ss_loss.update(loss_dict["ss"].mean().data.item())
        if loss_dict["sr"] is not None:
            ave_sr_loss.update(loss_dict["sr"].mean().data.item())
        if loss_dict["aff"] is not None:
            ave_aff_loss.update(loss_dict["aff"].mean().data.item())
        if loss_dict["psnr"] is not None:
            ave_psnr.update(loss_dict["psnr"].mean().data.item())

        # calculate accuracy, and display
        if i % cfg.TRAIN.disp_iter == 0:
            logger.info(f'Epoch: [{epoch}][{i}/{cfg.TRAIN.epoch_iters}], Time: {batch_time.average():.2f}, Data: {data_time.average():.2f}, '
                  f'lr_encoder: {cfg.TRAIN.running_lr_encoder:.6f}, lr_decoder: {cfg.TRAIN.running_lr_decoder:.6f}, '
                  f'Accuracy: {ave_acc.average():4.2f}, Loss: {ave_total_loss.average():.6f}, '
                  f'Loss_ss: {ave_ss_loss.average():.6f}, Loss_sr: {ave_sr_loss.average():.6f}, Loss_aff: {ave_aff_loss.average():.6f}, '
                  f'PSNR: {ave_psnr.average():.6f}'
            )

            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            if loss_dict["acc"] is not None:
                history['train']['acc'].append(loss_dict["acc"].mean().data.item())


def checkpoint(nets, history, cfg, epoch, logger):
    logger.info('Saving checkpoints...')
    (net_encoder, net_decoder_ss, net_decoder_sr, crit_ss, crit_sr, crit_aff) = nets

    dict_encoder = net_encoder.state_dict()

    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    
    if net_decoder_ss is not None:
        torch.save(
            net_decoder_ss.state_dict(),
            '{}/decoder_ss_epoch_{}.pth'.format(cfg.DIR, epoch))

    if net_decoder_sr is not None:
        torch.save(
            net_decoder_sr.state_dict(),
            '{}/decoder_sr_epoch_{}.pth'.format(cfg.DIR, epoch))

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, torchvision.models.swin_transformer.ShiftedWindowAttention):
            group_no_decay.append(m.relative_position_bias_table)
        elif isinstance(m, SAShiftedWindowAttention):
            group_no_decay.append(m.relative_position_bias_table)
        else:
            try:
                group_no_decay.append(m.weight)
            except:
                pass

            try:
                group_no_decay.append(m.bias)
            except:
                pass

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, cfg):
    (net_encoder, net_decoder_ss, net_decoder_sr, crit_ss, crit_sr, crit_aff) = nets
    if cfg.TRAIN.optim == "SGD":
        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        if net_decoder_ss is not None:
            optimizer_decoder_ss = torch.optim.SGD(
                group_weight(net_decoder_ss),
                lr=cfg.TRAIN.lr_decoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_decoder_ss = None
        if net_decoder_sr is not None:
            optimizer_decoder_sr = torch.optim.SGD(
                group_weight(net_decoder_sr),
                lr=cfg.TRAIN.lr_decoder,
                momentum=cfg.TRAIN.beta1,
                weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_decoder_sr = None
    elif cfg.TRAIN.optim == "AdamW":
        optimizer_encoder = torch.optim.AdamW(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            weight_decay=cfg.TRAIN.weight_decay)
        if net_decoder_ss is not None:
            optimizer_decoder_ss = torch.optim.AdamW(
                group_weight(net_decoder_ss),
                lr=cfg.TRAIN.lr_decoder,
                weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_decoder_ss = None
        if net_decoder_sr is not None:
            optimizer_decoder_sr = torch.optim.AdamW(
                group_weight(net_decoder_sr),
                lr=cfg.TRAIN.lr_decoder,
                weight_decay=cfg.TRAIN.weight_decay)
        else:
            optimizer_decoder_sr = None
    else:
        raise Exception('Unknown optimizer!')
    return (optimizer_encoder, optimizer_decoder_ss, optimizer_decoder_sr)


def adjust_learning_rate(optimizers, cur_iter, cfg):
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    (optimizer_encoder, optimizer_decoder_ss, optimizer_decoder_sr) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    if optimizer_decoder_ss is not None:
        for param_group in optimizer_decoder_ss.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_decoder
    if optimizer_decoder_sr is not None:
        for param_group in optimizer_decoder_sr.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus, logger):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    if cfg.MODEL.arch_decoder_ss != "":
        net_decoder_ss = ModelBuilder.build_decoder_ss(
            arch=cfg.MODEL.arch_decoder_ss.lower(),
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=cfg.MODEL.weights_decoder_ss)
    else:
        net_decoder_ss = None
    if cfg.MODEL.arch_decoder_sr != "":
        net_decoder_sr = ModelBuilder.build_decoder_sr(
            arch=cfg.MODEL.arch_decoder_sr.lower(),
            weights=cfg.MODEL.weights_decoder_sr)
    else:
        net_decoder_sr = None
    logger.info(f"Encoder arch:\n{net_encoder}")  
    logger.info(f"Decoder_ss arch:\n{net_decoder_ss}")
    logger.info(f"Decoder_sr arch:\n{net_decoder_sr}")

    # Loss Functions
    if cfg.MODEL.arch_decoder_ss != "":
        crit_ss = nn.NLLLoss(ignore_index=-1)
        logger.info(f"Decoder_ss loss: NLLLoss") 
    else:
        crit_ss = None
    if cfg.MODEL.arch_decoder_sr != "":
        crit_sr = nn.L1Loss()
        logger.info(f"Decoder_sr loss: L1Loss") 
    else:
        crit_sr = None
    if cfg.TRAIN.aff_loss in ["aa", "fa"]:
        crit_aff = nn.L1Loss()
        logger.info(f"Affinity loss: L1Loss") 
    else:
        crit_aff = None

    options = {
        "aff_loss": cfg.TRAIN.aff_loss,
        "w_1": cfg.TRAIN.w_1,
        "w_2": cfg.TRAIN.w_2,
        "w_3": cfg.TRAIN.w_3,
        "deep_sup_scale": cfg.TRAIN.deep_sup_scale if cfg.MODEL.arch_decoder_ss.endswith('deepsup') else None,
    }
    if "city" in cfg.DATASET.root_dataset:
        segmentation_module = SegmentationModuleCity(
            net_encoder, net_decoder_ss, net_decoder_sr, 
            crit_ss, crit_sr, crit_aff, options)
    else:
        segmentation_module = SegmentationModule(options=options)

    # Dataset and Loader
    if "city" in cfg.DATASET.root_dataset:
        dataset_train = TrainDatasetCity(
                cfg.DATASET.root_dataset,
                cfg.DATASET.list_train,
                cfg.DATASET,
                batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
    else:
        if not cfg.DATASET.square_crop:
            dataset_train = TrainDataset(
                cfg.DATASET.root_dataset,
                cfg.DATASET.list_train,
                cfg.DATASET,
                batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)
        else:
            dataset_train = TrainDatasetSquareCrop(
                cfg.DATASET.root_dataset,
                cfg.DATASET.list_train,
                cfg.DATASET,
                batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),  # we have modified data_parallel
        shuffle=False,  # we do not use this param
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    logger.info('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))

    # create loader iterator
    iterator_train = iter(loader_train)

    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder_ss, net_decoder_sr, crit_ss, crit_sr, crit_aff)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg, logger)

        # checkpointing
        checkpoint(nets, history, cfg, epoch+1, logger)

    logger.info('Training Done!')


if __name__ == '__main__':
    assert Version(torch.__version__) >= Version('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet18-ppm.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "--note",
        default=None,
        type=str,
        help="some notes of the experiment"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()
    
    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    log_count = len(glob.glob(os.path.join(cfg.DIR, f'train_*.log')))
    log_filename = os.path.join(cfg.DIR, f'train_{log_count}.log')
    logger = setup_logger(distributed_rank=0, filename=log_filename)   # TODO
    logger.info(f"This exp: {args.note}")
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder), "encoder checkpoint does not exitst!"
        if cfg.MODEL.arch_decoder_ss != "":
            cfg.MODEL.weights_decoder_ss = os.path.join(
                cfg.DIR, 'decoder_ss_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            assert os.path.exists(cfg.MODEL.weights_decoder_ss), "decoder_ss checkpoint does not exitst!"
        if cfg.MODEL.arch_decoder_sr != "":
            cfg.MODEL.weights_decoder_sr = os.path.join(
                cfg.DIR, 'decoder_sr_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            assert os.path.exists(cfg.MODEL.weights_decoder_sr), "decoder_sr checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu
    logger.info(f"Using gpus: {gpus}. Batch size per gpu: {cfg.TRAIN.batch_size_per_gpu}. Total batch size: {cfg.TRAIN.batch_size}.")

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus, logger)
