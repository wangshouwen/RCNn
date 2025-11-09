import os
import pickle
import argparse
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from utils.build_cfg import setup_cfg
from utils.helper import *
from utils.lr_scheduler import build_lr_scheduler
from datasets import Caption, build_dataset
from model import build_text_model
from utils.trainer_text import train, validate, validate_mssa


parser = argparse.ArgumentParser(description='PyTorch-Caption_Training')
parser.add_argument('--config_file', dest='config_file', type=str, help='model config file path')
parser.add_argument('--output_dir', type=str, default=None)


def main():
    global args
    args = parser.parse_args()
    cfg = setup_cfg(args)
    setup_seed(cfg.SEED)
    cudnn.benchmark = True

    record_name = datetime.datetime.now().strftime('%m-%d-%H:%M:%S') + "_" + "RENAME"
    record_path = os.path.join(cfg.OUTPUT_DIR, record_name)
    cfg.defrost()
    cfg.RECORD_PATH = record_path
    cfg.freeze()
    os.makedirs(record_path, exist_ok=True)
    logger = init_log(record_path)
    write_description_to_folder(os.path.join(record_path, "configs.txt"), cfg)

    # build train and test dataloaders
    train_dataset = Caption(text_path=cfg.DATASET.TEXT_PATH, dataset=cfg.DATASET.NAME, extra_template=True)
    test_dataset = build_dataset(cfg, cfg.DATASET.TEST_SPLIT)
    seen_cls_names = test_dataset.seen_names
    unseen_cls_names = test_dataset.unseen_names
    if isinstance(seen_cls_names, np.ndarray):
        seen_cls_names = seen_cls_names.tolist()
        unseen_cls_names = unseen_cls_names.tolist()
    all_cls_names = seen_cls_names + unseen_cls_names
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                              shuffle=cfg.DATALOADER.TRAIN_X.SHUFFLE, num_workers=cfg.DATALOADER.NUM_WORKERS, 
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                             shuffle=cfg.DATALOADER.TEST.SHUFFLE, num_workers=cfg.DATALOADER.NUM_WORKERS,
                             pin_memory=True, drop_last=False)
    
    #build model
    model = build_text_model(cfg, all_cls_names)
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    try:
        param_group = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
                param_group.append(param)
    except:
        param_group = model.module.parameters()
    optim = torch.optim.SGD(param_group, lr=cfg.OPTIM.LR, momentum=cfg.OPTIM.MOMENTUM, weight_decay=cfg.OPTIM.WEIGHT_DECAY)
    sched = build_lr_scheduler(optim, cfg.OPTIM)

    best_unseen_F1 = 0
    best_gzsl_F1 = 0
    start_epoch = 0
    if cfg.RESUME is not None:
        if os.path.exists(cfg.RESUME):
            logger.info('... loading pretrained weights from %s' % cfg.RESUME)
            checkpoint = torch.load(cfg.RESUME, map_location='cpu')
            start_epoch = checkpoint['epoch']
            best_unseen_F1 = checkpoint['best_unseen_F1']
            best_gzsl_F1 = checkpoint['best_gzsl_F1']
            model.prompt_learner.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            sched.load_state_dict(checkpoint['scheduler'])
    
    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        batch_time, losses = train(train_loader, model, optim, sched, cfg, logger, epoch)
        logger.info('Train: [{0}/{1}]\t'
                    'Time {batch_time.avg:.3f}\t'
                    'Loss {losses.avg:.2f}\t'.format(
                    epoch+1, cfg.OPTIM.MAX_EPOCH, batch_time=batch_time, losses=losses))
        
        if (epoch+1) % cfg.TRAIN.EVAL_PERIOD == 0 or epoch == cfg.OPTIM.MAX_EPOCH - 1:
            if cfg.cutimage is False:
                f1_unseen, f1_gzsl = validate(test_loader, model, cfg, logger, epoch)
            else:
                f1_unseen, f1_gzsl = validate_mssa(test_loader, model, cfg, logger, epoch)

            is_unseen_best = f1_unseen > best_unseen_F1
            if is_unseen_best:
                best_unseen_F1 = f1_unseen

            is_gzsl_best = f1_gzsl > best_gzsl_F1
            if is_gzsl_best:
                best_gzsl_F1 = f1_gzsl

            save_dict = {'epoch': epoch + 1,
                         'state_dict': model.prompt_learner.state_dict(),
                         'best_unseen_F1': best_unseen_F1,
                         'best_gzsl_F1': best_unseen_F1,
                         'optimizer': optim.state_dict(),
                         'scheduler': sched.state_dict()
                         }
            save_checkpoint(save_dict, is_unseen_best, record_path, prefix='unseen')
            save_checkpoint(save_dict, is_gzsl_best, record_path, prefix='gzsl')
            logger.info(' * best_F1_3_unseen={best1:.4f}\t''best_F1_3_global={best2:.4f}'.format(best1=best_unseen_F1, best2=best_gzsl_F1))


if __name__ == '__main__':
    main()





    