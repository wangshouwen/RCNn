import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from utils.helper import AverageMeter, compute_AP, compute_F1
from torch.cuda.amp import autocast

def no_similar_loss(x):
    similar = x @ x.t()
    mask = 1 - torch.eye(x.size(0))
    mask = mask.to(x.device)
    dist = (1 + similar) * mask
    return dist.sum()

    
def train(data_loader, model, optim, sched, cfg, logger, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    model.train()
    model.text_encoder.eval()

    end = time.time()
    for i, (captions, targets) in enumerate(data_loader):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        captions = captions.to(device)
        targets = targets.to(device)

        with autocast():
            text_features, cls_embeddings = model(captions)
        clsname_cur = cls_embeddings[targets]

        loss = F.mse_loss(clsname_cur, text_features, reduction='sum')
        if cfg.no_sim is True:
            loss1 = no_similar_loss(cls_embeddings)
            loss = loss + cfg.no_sim_weights * loss1
            
        optim.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        losses.update(loss.item(), captions.size(0))
        batch_time.update(time.time()-end)
        end = time.time()

        if i % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info('Train: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t \t'
                  'Loss {losses.val:.2f} ({losses.avg:.2f})'.format(
                str(i).rjust(4), len(data_loader), batch_time=batch_time, losses=losses))
    if sched is not None:
        sched.step()
    return batch_time, losses

def validate(data_loader, model, cfg, logger, epoch):
    model.eval()
    preds_unseen = []
    preds_glob = []
    labels_unseen = []
    labels_glob = []

    with torch.no_grad():
        with autocast():
            text_feats = model.encode_text_features()
    
    for images, seen_labels, unseen_labels in tqdm(data_loader):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        images = images.to(device)
        num_seen_cls = seen_labels.size(1)
        
        with torch.no_grad():
            with autocast():
                output = model.forward_for_open(images, text_feats)
            logits_unseen = output[:, num_seen_cls:]
            logits_glob = output
            
        preds_unseen.append(logits_unseen.float().cpu())
        preds_glob.append(logits_glob.float().cpu())
        labels_unseen.append(unseen_labels)
        tmp_labels = torch.cat([seen_labels, unseen_labels], dim=1)
        labels_glob.append(tmp_labels)

    preds_unseen = torch.cat(preds_unseen, dim=0)
    preds_glob = torch.cat(preds_glob, dim=0)
    labels_unseen = torch.cat(labels_unseen, dim=0)
    labels_glob = torch.cat(labels_glob, dim=0)
    logger.info("Evaluating predictions over all images")

    unseen_mask = (labels_unseen > 0).sum(1) > 0
    labels_unseen = labels_unseen[unseen_mask]
    preds_unseen = preds_unseen[unseen_mask]
    preds_unseen_cp = preds_unseen.clone()
    ap_unseen = compute_AP(preds_unseen, labels_unseen)
    mAP_unseen = 100 * ap_unseen.mean()
    F1_3_u, P_3_u, R_3_u = compute_F1(preds_unseen, labels_unseen, k_val=3)
    F1_5_u, P_5_u, R_5_u = compute_F1(preds_unseen_cp, labels_unseen, k_val=5)

    logger.info('Test: [{}/{}] \t mAP_unseen {:.2f}\t \n'
                ' P_3_unseen {:.4f} \t R_3_unseen {:.4f} \t F1_3_unseen {:.4f}\t \n'
                ' P_5_unseen {:.4f} \t R_5_unseen {:.4f} \t F1_5_unseen {:.4f} \t '.format(
                epoch+1, cfg.OPTIM.MAX_EPOCH, mAP_unseen, P_3_u, R_3_u, F1_3_u, P_5_u, R_5_u, F1_5_u))
    
    glob_mask = (labels_glob > 0).sum(1) > 0
    labels_glob = labels_glob[glob_mask]
    preds_glob = preds_glob[glob_mask]
    preds_glob_cp = preds_glob.clone()
    ap_glob = compute_AP(preds_glob, labels_glob)
    mAP_glob = 100 * ap_glob.mean()
    F1_3_g, P_3_g, R_3_g = compute_F1(preds_glob, labels_glob, k_val=3)
    F1_5_g, P_5_g, R_5_g = compute_F1(preds_glob_cp, labels_glob, k_val=5)

    logger.info('Test: [{}/{}] \t mAP_glob {:.2f}\t \n'
                ' P_3_glob {:.4f} \t R_3_glob {:.4f} \t F1_3_glob {:.4f}\t \n'
                ' P_5_glob {:.4f} \t R_5_glob {:.4f} \t F1_5_glob {:.4f} \t '.format(
                epoch+1, cfg.OPTIM.MAX_EPOCH, mAP_glob, P_3_g, R_3_g, F1_3_g, P_5_g, R_5_g, F1_5_g))
    
    return F1_3_u, F1_3_g

def validate_mssa(data_loader, model, cfg, logger, epoch):
    model.eval()
    preds_unseen = []
    preds_glob = []
    labels_unseen = []
    labels_glob = []

    with torch.no_grad():
        with autocast():
            text_feats = model.encode_text_features()
    
    for images, seen_labels, unseen_labels, img_list in tqdm(data_loader):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        images = images.to(device)
        num_seen_cls = seen_labels.size(1)
        
        output_list = []
        with torch.no_grad():
            with autocast():
                output = model.forward_for_test(images, text_feats)
                for img in img_list:
                    output_ = model.forward_for_test(img.to(device), text_feats)
                    output_list.append(output_)
                output_final = model.aggregatorplus(output, output_list, count_list=cfg.count_list)

        logits_unseen = output_final[:, num_seen_cls:]
        logits_glob = output_final

        preds_unseen.append(logits_unseen.float().cpu())
        preds_glob.append(logits_glob.float().cpu())
        labels_unseen.append(unseen_labels)
        tmp_labels = torch.cat([seen_labels, unseen_labels], dim=1)
        labels_glob.append(tmp_labels)

    preds_unseen = torch.cat(preds_unseen, dim=0)
    preds_glob = torch.cat(preds_glob, dim=0)
    labels_unseen = torch.cat(labels_unseen, dim=0)
    labels_glob = torch.cat(labels_glob, dim=0)
    logger.info("Evaluating predictions over all images")

    unseen_mask = (labels_unseen > 0).sum(1) > 0
    labels_unseen = labels_unseen[unseen_mask]
    preds_unseen = preds_unseen[unseen_mask]
    preds_unseen_cp = preds_unseen.clone()
    ap_unseen = compute_AP(preds_unseen, labels_unseen)
    mAP_unseen = 100 * ap_unseen.mean()
    F1_3_u, P_3_u, R_3_u = compute_F1(preds_unseen, labels_unseen, k_val=3)
    F1_5_u, P_5_u, R_5_u = compute_F1(preds_unseen_cp, labels_unseen, k_val=5)

    logger.info('Test: [{}/{}] \t mAP_unseen {:.2f}\t \n'
                ' P_3_unseen {:.4f} \t R_3_unseen {:.4f} \t F1_3_unseen {:.4f}\t \n'
                ' P_5_unseen {:.4f} \t R_5_unseen {:.4f} \t F1_5_unseen {:.4f} \t '.format(
                epoch+1, cfg.OPTIM.MAX_EPOCH, mAP_unseen, P_3_u, R_3_u, F1_3_u, P_5_u, R_5_u, F1_5_u))
    
    glob_mask = (labels_glob > 0).sum(1) > 0
    labels_glob = labels_glob[glob_mask]
    preds_glob = preds_glob[glob_mask]
    preds_glob_cp = preds_glob.clone()
    ap_glob = compute_AP(preds_glob, labels_glob)
    mAP_glob = 100 * ap_glob.mean()
    F1_3_g, P_3_g, R_3_g = compute_F1(preds_glob, labels_glob, k_val=3)
    F1_5_g, P_5_g, R_5_g = compute_F1(preds_glob_cp, labels_glob, k_val=5)

    logger.info('Test: [{}/{}] \t mAP_glob {:.2f}\t \n'
                ' P_3_glob {:.4f} \t R_3_glob {:.4f} \t F1_3_glob {:.4f}\t \n'
                ' P_5_glob {:.4f} \t R_5_glob {:.4f} \t F1_5_glob {:.4f} \t '.format(
                epoch+1, cfg.OPTIM.MAX_EPOCH, mAP_glob, P_3_g, R_3_g, F1_3_g, P_5_g, R_5_g, F1_5_g))
    
    return F1_3_u, F1_3_g