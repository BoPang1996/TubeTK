import argparse
import os
import torch
from tensorboardX import SummaryWriter
from network.tubetk import TubeTK
from dataset.dataLoader import Data_Loader_MOT
from optim.solver import make_optimizer as makeOpt
from configs.default import __C, cfg_from_file
from utils.util import AverageMeter
from tqdm import tqdm
from optim.lr_scheduler import WarmupMultiStepLR
import warnings
import numpy as np
try:
    from apex import amp
    import apex
except:
    pass
warnings.filterwarnings('ignore')


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.half()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not torch.distributed.is_available():
        return
    if not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return
    torch.distributed.barrier()


def print_dict(string, rank):
    if rank == 0:
        print(string)


def run_one_iter(model, optimizer, data, scheduler, test):
    imgs, img_metas, gt_tubes, gt_labels, start_frame = data

    # =================================Visualization================================================
    # vis_input(imgs, img_metas, gt_bboxes, gt_labels, start_frame, stride=model_arg.frame_stride, out_folder='/home/pb/results/')
    # ==============================================================================================

    # Get Input
    imgs = imgs.cuda()
    for i in range(len(gt_tubes)):
        gt_tubes[i] = gt_tubes[i].cuda()
        gt_labels[i] = gt_labels[i].cuda()

    if not test:
        scheduler.step()

    # Forward
    if not test:
        losses = model(imgs, img_metas, return_loss=True, gt_tubes=gt_tubes, gt_labels=gt_labels)
        res = losses
    else:
        with torch.no_grad():
            bbox_list = model(imgs, img_metas, return_loss=False, gt_tubes=gt_tubes, gt_labels=gt_labels)
        bbox_list[:, :, 0] += start_frame
        res = bbox_list

    # Backward
    if not test:
        if losses:
            optimizer.zero_grad()
            loss = torch.zeros(1).cuda()
            for l in losses:
                if 'loss_cls' in l:
                    loss += 1e3 * losses[l]
                else:
                    loss += losses[l]
            if not train_arg.apex:
                loss.backward()
            else:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            optimizer.step()

    return res


def train(model, optimizer, data_loader, scheduler, writer, max_acc=0, step_start=0):
    loss_cls_accumulate = AverageMeter()
    loss_reg_accumulate = AverageMeter()
    loss_center_accumulate = AverageMeter()
    max_acc = max_acc

    loader = data_loader.train_loader
    model.train()
    if train_arg.apex:
        model.apply(fix_bn)
    if train_arg.rank == 0:
        loader = tqdm(loader, ncols=20)

    loader_len = len(loader)
    for step, data in enumerate(loader):
        # Input
        if step > loader_len - step_start:
            break
        step += step_start
        losses = run_one_iter(model, optimizer, data, scheduler, False)

        # Loss and results
        if losses:
            if not np.isnan(losses['loss_cls'].data.cpu().numpy()):
                loss_cls_accumulate.update(val=losses['loss_cls'].data.cpu().numpy())
            if not np.isnan(losses['loss_reg'].data.cpu().numpy()):
                loss_reg_accumulate.update(val=losses['loss_reg'].data.cpu().numpy())
            if not np.isnan(losses['loss_centerness'].data.cpu().numpy()):
                loss_center_accumulate.update(val=losses['loss_centerness'].data.cpu().numpy())

        if train_arg.rank == 0:
            writer.add_scalar('train/loss_cls', loss_cls_accumulate.avg, step)
            writer.add_scalar('train/loss_reg', loss_reg_accumulate.avg, step)
            writer.add_scalar('train/loss_center', loss_center_accumulate.avg, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]["lr"], step)

        if step % 1000 == 999:
            if train_arg.rank == 0:
                print('save model')
                torch.save({'state': model.state_dict(),
                            'max_acc': max_acc,
                            'step': step,
                            'opt': optimizer.state_dict(),
                            'sched': scheduler.state_dict()},
                           train_arg.model_path + '/' + train_arg.model_name)

        if step % train_arg.reset_iter == train_arg.reset_iter - 1:
            loss_cls_accumulate.reset()
            loss_reg_accumulate.reset()
            loss_center_accumulate.reset()

        if train_arg.local_rank == 0:
            loader.set_description('Loss_cls: ' + str(loss_cls_accumulate.avg)[0:6] +
                                   ',\tLoss_reg: ' + str(loss_reg_accumulate.avg)[0:6] +
                                   ',\tLoss_center: ' + str(loss_center_accumulate.avg)[0:6], refresh=False)


def main(train_arg, model_arg):
    torch.distributed.init_process_group(backend="nccl", init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    print('Rank: ' + str(train_arg.rank) + " Start!")
    torch.cuda.set_device(local_rank)

    print_dict("Building TubeTK Model", train_arg.local_rank)
    model = TubeTK(num_classes=1, arg=model_arg, pretrained=True)

    data_loader = Data_Loader_MOT(
        batch_size=train_arg.batch_size,
        num_workers=8,
        input_path=train_arg.data_url,
        train_epoch=train_arg.epochs,
        model_arg=model_arg,
        dataset=train_arg.dataset,
        test_epoch=1
    )
    # =================================Visualization================================================
    # loader = data_loader.train_loader
    # for step, data in enumerate(loader):
    #     imgs, img_metas, gt_bboxes, gt_labels, start_frame = data
    #
    #     vis_input(imgs, img_metas, gt_bboxes, gt_labels, start_frame, stride=model_arg.frame_stride,
    #               out_folder='/home/pb/results/')
    # ==============================================================================================

    model = model.cuda(local_rank)
    optimizer = makeOpt(train_arg, model)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if train_arg.apex:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O1',
                                          # loss_scale='dynamic',
                                          # keep_batchnorm_fp32=False
                                          )

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    sched = WarmupMultiStepLR(
        optimizer,
        milestones=train_arg.mileStone,
        warmup_factor=0.1,
        warmup_iters=0,
        warmup_method='linear')

    max_acc = 0
    step = 0

    if train_arg.resume:
        print_dict("Loading Model", train_arg.local_rank)
        checkpoint = torch.load(train_arg.model_path + '/' + train_arg.model_name, map_location=
                                {'cuda:0': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:1': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:2': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:3': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:4': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:5': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:6': 'cuda:' + str(train_arg.local_rank),
                                 'cuda:7': 'cuda:' + str(train_arg.local_rank)})
        model.load_state_dict(checkpoint['state'], strict=False)
        optimizer.load_state_dict(checkpoint['opt'])
        sched.load_state_dict(checkpoint['sched'])
        sched.milestones = train_arg.mileStone
        step = checkpoint['step'] + 1
        sched.last_epoch = step
        max_acc = checkpoint['max_acc']
        print_dict("Finish Loading", train_arg.local_rank)
        del checkpoint

    if train_arg.rank == 0:
        tensorboard_writer = SummaryWriter(train_arg.logName, purge_step=step)
    else:
        tensorboard_writer = None

    print_dict("Training", train_arg.local_rank)
    train(model, optimizer, data_loader, sched, tensorboard_writer, max_acc=max_acc, step_start=step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Sub-JHMDB rgb frame training')
    parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')
    parser.add_argument('--mileStone', nargs='+', type=int, default=[7500, 15000], help='mileStone for lr Sched')
    parser.add_argument('--reset_iter', default=200, type=list, help='test iter')
    parser.add_argument('--model_path', default='./models', type=str, help='model path')
    parser.add_argument('--model_name', default='TubeTK', type=str, help='model name')
    parser.add_argument('--data_url', default='./data/', type=str, help='data path')
    parser.add_argument('--dataset', default='MOT17', type=str, help='MOT17, JTA, MOTJTA')

    parser.add_argument('--config', default=None, type=str, help='config file')

    parser.add_argument('--logName', type=str,
                        default='./logs/TubeTK_log', help='log dir name')

    parser.add_argument('--local_rank', type=int, help='gpus')

    parser.add_argument('--resume', action='store_true', help='whether resume')

    parser.add_argument('--apex', action='store_true', help='whether use apex')

    train_arg, unparsed = parser.parse_known_args()

    model_arg = __C
    if train_arg.config is not None:
        cfg_from_file(train_arg.config)

    train_arg.rank = int(os.environ["RANK"])
    if train_arg.rank == 0:
        try:
            os.makedirs(train_arg.model_path)
        except:
            pass

    main(train_arg, model_arg)
