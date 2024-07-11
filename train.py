import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from visdom import Visdom

from dataset.augmentation import get_transform
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
from scheduler.cos_annealing_with_restart import CosineAnnealingLR_with_Restart
from scheduler.cosine_lr import CosineLRScheduler
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from batch_engine import valid_trainer, batch_trainer
from dataset.pedes_attr.pedes import PedesAttr
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone
from models.backbone import resnet
from losses import bceloss, scaledbceloss
from models import base_block
from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool
import yaml


# torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)


def main(args):
    set_seed(605)
    with open(args.cfg, 'r') as file:
        prime_service = yaml.safe_load(file)


    exp_dir = os.path.join('exp_result', prime_service['DATASET']['NAME'])

    model_dir, log_dir = get_model_log_path(exp_dir, prime_service['NAME'])
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    visdom = None
    if prime_service['VIS']['VISDOM']:
        visdom = Visdom(env=f"{prime_service['DATASET']['NAME']}_" + prime_service['NAME'], port=8401)
        assert visdom.check_connection()

    writer = None
    if prime_service['VIS']['TENSORBOARD']['ENABLE']:
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        writer_dir = os.path.join(exp_dir, prime_service['NAME'], 'runs', current_time)
        writer = SummaryWriter(log_dir=writer_dir)

    if prime_service['REDIRECTOR']:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    """
    the reason for args usage is CfgNode is immutable
    """

    args.distributed = None

    args.world_size = 1
    args.rank = 0  # global rank



    train_tsfm, valid_tsfm = get_transform(prime_service)
    if args.local_rank == 0:
        print(train_tsfm)

    train_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['TRAIN_SPLIT'], transform=train_tsfm,
                            target_transform=prime_service['DATASET']['TARGETTRANSFORM'])

    valid_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['VAL_SPLIT'], transform=valid_tsfm,
                            target_transform=prime_service['DATASET']['TARGETTRANSFORM'])
    
    test_set = PedesAttr(cfg=prime_service, split=prime_service['DATASET']['TEST_SPLIT'], transform=valid_tsfm,
                            target_transform=prime_service['DATASET']['TARGETTRANSFORM'])


    train_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=prime_service['TRAIN']['BATCH_SIZE'],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=prime_service['TRAIN']['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=prime_service['TRAIN']['BATCH_SIZE'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"train -> {len(train_loader)*64}")
    print(f"valid -> {len(valid_loader)*64}")
    print(f"test  -> {len(test_loader)*64}")

    if args.local_rank == 0:
        print('-' * 60)
        print(f"{prime_service['DATASET']['NAME']} attr_num : {train_set.attr_num}, eval_attr_num : {train_set.eval_attr_num} "
              f"{prime_service['DATASET']['TRAIN_SPLIT']} set: {len(train_loader.dataset)}, "
              f"{prime_service['DATASET']['TEST_SPLIT']} set: {len(valid_loader.dataset)}, "
              f"{prime_service['DATASET']['TEST_SPLIT']} set: {len(test_loader.dataset)}, "
              )

    labels = train_set.label
    label_ratio = labels.mean(0) if prime_service['LOSS']['SAMPLE_WEIGHT'] else None

    backbone, c_output = build_backbone(prime_service['BACKBONE']['TYPE'], prime_service['BACKBONE']['MULTISCALE'])


    classifier = build_classifier(prime_service['CLASSIFIER']['NAME'])(
        nattr=train_set.attr_num,
        c_in=c_output,
        bn=prime_service['CLASSIFIER']['BN'],
        pool=prime_service['CLASSIFIER']['POOLING'],
        scale =prime_service['CLASSIFIER']['SCALE']
    )

    model = FeatClassifier(backbone, classifier, bn_wd=prime_service['TRAIN']['BN_WD'])
    if args.local_rank == 0:
        print(f"backbone: {prime_service['BACKBONE']['TYPE']}, classifier: {prime_service['CLASSIFIER']['NAME']}")
        print(f"model_name: {prime_service['NAME']}")


    model = model.cuda()
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model)

    model_ema = None
    if prime_service['TRAIN']['EMA']['ENABLE']:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=prime_service['TRAIN']['EMA']['DECAY'], device='cpu' if prime_service['TRAIN']['EMA']['FORCE_CPU'] else None)

    if prime_service['RELOAD']['TYPE']:
        model = get_reload_weight(model_dir, model, prime_service,pth=prime_service['RELOAD']['PTH'])

    loss_weight = prime_service['LOSS']['LOSS_WEIGHT']


    criterion = build_loss(prime_service['LOSS']['TYPE'])(
        sample_weight=label_ratio, scale=prime_service['CLASSIFIER']['SCALE'], size_sum=prime_service['LOSS']['SIZESUM'], tb_writer=writer)
    criterion = criterion.cuda()

    if prime_service['TRAIN']['BN_WD']:
        param_groups = [{'params': model.module.finetune_params(),
                         'lr': prime_service['TRAIN']['LR_SCHEDULER']['LR_FT'],
                         'weight_decay': prime_service['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']},
                        {'params': model.module.fresh_params(),
                         'lr': prime_service['TRAIN']['LR_SCHEDULER']['LR_NEW'],
                         'weight_decay': prime_service['TRAIN']['OPTIMIZER']['WEIGHT_DECAY']}]
    else:
        # bn parameters are not applied with weight decay
        ft_params = seperate_weight_decay(
            model.module.finetune_params(),
            lr=prime_service['TRAIN']['LR_SCHEDULER']['LR_FT'],
            weight_decay=prime_service['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])

        fresh_params = seperate_weight_decay(
            model.module.fresh_params(),
            lr=prime_service['TRAIN']['LR_SCHEDULER']['LR_NEW'],
            weight_decay=prime_service['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])

        param_groups = ft_params + fresh_params

    if prime_service['TRAIN']['OPTIMIZER']['TYPE'].lower() == 'sgd':
        optimizer = torch.optim.SGD(param_groups, momentum=prime_service['TRAIN']['OPTIMIZER']['MOMENTUM'])
    elif prime_service['TRAIN']['OPTIMIZER']['TYPE'].lower() == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    elif prime_service['TRAIN']['OPTIMIZER']['TYPE'].lower() == 'adamw':
        optimizer = AdamW(param_groups)
    else:
        assert None, f"{prime_service['TRAIN']['OPTIMIZER']['TYPE']} is not implemented"

    if prime_service['TRAIN']['LR_SCHEDULER']['TYPE'] == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)
        if prime_service['CLASSIFIER']['BN']:
            assert False, 'BN can not compatible with ReduceLROnPlateau'
    elif prime_service['TRAIN']['LR_SCHEDULER']['TYPE'] == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, milestones=prime_service['TRAIN']['LR_SCHEDULER']['LR_STEP'], gamma=0.1)
    elif prime_service['TRAIN']['LR_SCHEDULER']['TYPE'] == 'annealing_cosine':
        lr_scheduler = CosineAnnealingLR_with_Restart(
            optimizer,
            T_max=(prime_service['TRAIN']['MAX_EPOCH'] + 5) * len(train_loader),
            T_mult=1,
            eta_min=prime_service['TRAIN']['LR_SCHEDULER']['LR_NEW'] * 0.001
    )
    elif prime_service['TRAIN']['LR_SCHEDULER']['TYPE'] == 'warmup_cosine':


        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=prime_service['TRAIN']['MAX_EPOCH'],
            lr_min=1e-5,  # cosine lr 最终回落的位置
            warmup_lr_init=1e-4,
            warmup_t=prime_service['TRAIN']['MAX_EPOCH'] * prime_service['TRAIN']['LR_SCHEDULER']['WMUP_COEF'],
        )

    else:
        assert False, f"{prime_service['LR_SCHEDULER']['TYPE']} has not been achieved yet"
    best_metric, epoch = trainer(prime_service, epoch=prime_service['TRAIN']['MAX_EPOCH'],
                                 model=model, model_ema=model_ema,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 test_loader=test_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path,
                                 loss_w=loss_weight,
                                 viz=visdom,
                                 tb_writer=writer)
        

def trainer(cfg, epoch, model, model_ema, train_loader, valid_loader, test_loader,criterion, optimizer, lr_scheduler,
            path, loss_w, viz, tb_writer):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    result_path = path
    result_path = result_path.replace('ckpt_max', 'metric')
    result_path = result_path.replace('pth', 'pkl')

    for e in range(epoch):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        lr = optimizer.param_groups[1]['lr']

        train_loss, train_gt, train_probs, train_imgs, train_logits, train_loss_mtr = batch_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model,
            model_ema=model_ema,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            loss_w=loss_w,
            scheduler=lr_scheduler if cfg['TRAIN']['LR_SCHEDULER']['TYPE'] == 'annealing_cosine' else None,
        )


        valid_loss, valid_gt, valid_probs, valid_imgs, valid_logits, valid_loss_mtr = valid_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model_ema.module,
            valid_loader=valid_loader,
            criterion=criterion,
            loss_w=loss_w
        )

        test_loss, test_gt, test_probs, test_imgs, test_logits, test_loss_mtr = valid_trainer(
            cfg,
            args=args,
            epoch=e,
            model=model_ema.module,
            valid_loader=test_loader,
            criterion=criterion,
            loss_w=loss_w
        )


        if cfg['TRAIN']['LR_SCHEDULER']['TYPE'] == 'plateau':
            lr_scheduler.step(metrics=valid_loss)
        elif cfg['TRAIN']['LR_SCHEDULER']['TYPE'] == 'warmup_cosine':
            lr_scheduler.step(epoch=e + 1)
        elif cfg['TRAIN']['LR_SCHEDULER']['TYPE'] == 'multistep':
            lr_scheduler.step()

        train_result = get_pedestrian_metrics(train_gt, train_probs, index=None, cfg=cfg)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs, index=None, cfg=cfg)
        test_result = get_pedestrian_metrics(test_gt, test_probs, index=None, cfg=cfg)


        if args.local_rank == 0:
            print(f'Evaluation on train set, train losses {train_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        train_result.ma, np.mean(train_result.label_f1),
                        np.mean(train_result.label_pos_recall),
                        np.mean(train_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        train_result.instance_acc, train_result.instance_prec, train_result.instance_recall,
                        train_result.instance_f1))

            print(f'Evaluation on valid set, valid losses {valid_loss}\n',
                    'ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                        valid_result.ma, np.mean(valid_result.label_f1),
                        np.mean(valid_result.label_pos_recall),
                        np.mean(valid_result.label_neg_recall)),
                    'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                        valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                        valid_result.instance_f1))

            print(f'{time_str()}')
            print('-' * 60)

        cur_metric = valid_result.ma
        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = e
            save_ckpt(model_ema.module, path, e, maximum)

        result_list[e] = {
            'ma': test_result.ma,  # 'train_map': train_map,
            'label_f1': np.mean(test_result.label_f1),  # 'valid_map': valid_map,
            'pos_recall': np.mean(test_result.label_pos_recall), 
            'neg_recall': np.mean(test_result.label_neg_recall), 
            'Acc': test_result.instance_acc, 
            'Prec': test_result.instance_prec,
            'Rec': test_result.instance_recall,
            'F1': test_result.instance_f1
        }

        with open(result_path, 'wb') as f:
            pickle.dump(result_list, f)

    print("=====================================================")
    print(f"{cfg['NAME']},  best_metrc : {maximum} in epoch{best_epoch}")

    print(f"Evaluation on test set\n",
            "ma: {:.4f}, label_f1: {:.4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n".format(
                result_list[best_epoch]['ma'], np.mean(result_list[best_epoch]['label_f1']),
                np.mean(result_list[best_epoch]['pos_recall']),
                np.mean(result_list[best_epoch]['neg_recall'])),
            "Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}".format(
                result_list[best_epoch]['Acc'], result_list[best_epoch]['Prec'], result_list[best_epoch]['Rec'],
                result_list[best_epoch]['F1']))
    print("=====================================================")


    return maximum, best_epoch


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="./configs/pedes_baseline/pa100k.yaml",

    )

    parser.add_argument("--debug", type=str2bool, default="true")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    main(args)
