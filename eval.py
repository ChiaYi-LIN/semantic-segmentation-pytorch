# System libs
import os
import glob
import time
import argparse
from packaging.version import Version
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from mit_semseg.config import cfg
from mit_semseg.dataset import ValDataset
from mit_semseg.dataset_city import ValDatasetCity
from mit_semseg.models import ModelBuilder, SegmentationModule, SegmentationModuleCity
from mit_semseg.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from mit_semseg.lib.utils.calculate_psnr_ssim import Postprocess
from PIL import Image
from tqdm import tqdm

colors = loadmat('data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu, logger):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()
    postprocess = Postprocess()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            # if segSize[0] > 900 or segSize[1] > 900:
            #     pbar.update(1)
            #     continue

            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)
            reconstructs = torch.zeros(1, 3, segSize[0], segSize[1])
            reconstructs = async_copy_to(reconstructs, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                output_dict = segmentation_module(feed_dict, segSize=segSize)
                scores_tmp = output_dict.get("segment", None)
                reconstructs_tmp = output_dict.get("reconstruct", None)

                if scores_tmp is not None:
                    scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)
                if reconstructs_tmp is not None:
                    reconstructs = reconstructs + reconstructs_tmp / len(cfg.DATASET.imgSizes)
            
            if scores_tmp is not None:
                _, pred = torch.max(scores, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
            if reconstructs_tmp is not None:
                rgb_gt = batch_data['img_ori']
                rgb_pred = np.array(postprocess.inverse2pil(reconstructs[0]))
                assert (rgb_gt.shape[0] == segSize[0] and rgb_gt.shape[1] == segSize[1])
                assert (rgb_pred.shape[0] == segSize[0] and rgb_pred.shape[1] == segSize[1])
                psnr = postprocess.psnr(rgb_gt, rgb_pred)
                psnr_meter.update(psnr)

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        if scores_tmp is not None:
            acc, pix = accuracy(pred, seg_label)
            intersection, union = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

            # visualization
            if cfg.VAL.visualize:
                visualize_result(
                    (batch_data['img_ori'], seg_label, batch_data['info']),
                    pred,
                    os.path.join(cfg.DIR, 'result')
                )

        pbar.update(1)

    # summary
    if scores_tmp is not None:
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        for i, _iou in enumerate(iou):
            logger.info('class [{}], IoU: {:.4f}'.format(i, _iou))

        logger.info('[Eval Summary]:')
        logger.info(f'Mean IoU: {iou.mean():.4f}, Accuracy: {acc_meter.average()*100:.2f}%, PSNR: {psnr_meter.average():.4f}, Inference Time: {time_meter.average():.4f}s')
    else:
        logger.info('[Eval Summary]:')
        logger.info(f'PSNR: {psnr_meter.average():.4f}, Inference Time: {time_meter.average():.4f}s')

def main(cfg, gpu, logger):
    torch.cuda.set_device(gpu)

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
            weights=cfg.MODEL.weights_decoder_ss,
            use_softmax=True)
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

    if "city" in cfg.DATASET.root_dataset:
        segmentation_module = SegmentationModuleCity(net_encoder, net_decoder_ss, net_decoder_sr)
    else:
        segmentation_module = SegmentationModule(net_encoder, net_decoder_ss, net_decoder_sr)

    # Dataset and Loader
    if "city" in cfg.DATASET.root_dataset:
        dataset_val = ValDatasetCity(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET)
    else:
        dataset_val = ValDataset(
            cfg.DATASET.root_dataset,
            cfg.DATASET.list_val,
            cfg.DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu, logger)

    logger.info('Evaluation Done!')


if __name__ == '__main__':
    assert Version(torch.__version__) >= Version('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet18-ppm.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        help="gpu to use"
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

    log_count = len(glob.glob(os.path.join(cfg.DIR, f'eval_*.log')))
    log_filename = os.path.join(cfg.DIR, f'eval_{log_count}.log')
    logger = setup_logger(distributed_rank=0, filename=log_filename)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder), "encoder checkpoint does not exitst!"
    if cfg.MODEL.arch_decoder_ss != "":
        cfg.MODEL.weights_decoder_ss = os.path.join(
            cfg.DIR, 'decoder_ss_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_decoder_ss), "decoder_ss checkpoint does not exitst!"
    if cfg.MODEL.arch_decoder_sr != "":
        cfg.MODEL.weights_decoder_sr = os.path.join(
            cfg.DIR, 'decoder_sr_' + cfg.VAL.checkpoint)
        assert os.path.exists(cfg.MODEL.weights_decoder_sr), "decoder_sr checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    main(cfg, args.gpu, logger)
