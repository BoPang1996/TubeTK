import pickle
import os
import numpy as np
import torch
import warnings
from tqdm import tqdm
from Metrics import evaluateTracking
from dataset.dataLoader import Data_Loader_MOT
from network.tubetk import TubeTK
from post_processing.tube_nms import multiclass_nms
from apex import amp
import argparse
import multiprocessing
from configs.default import __C, cfg_from_file
from post_processing.tube_iou_matching import matching
warnings.filterwarnings('ignore')
import shutil


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


def match_video(video_name, tmp_dir, output_dir, model_arg):
    tubes_path = os.path.join(tmp_dir, video_name)
    tubes = []
    frames = sorted([int(x) for x in os.listdir(tubes_path)])
    for f in frames:
        tube = pickle.load(open(os.path.join(tubes_path, str(f)), 'rb'))
        tubes.append(tube)

    tubes = np.concatenate(tubes)
    matching(tubes, save_path=os.path.join(output_dir, video_name + '.txt'), verbose=True, arg=model_arg)


def evaluate(model, loader, test_arg, model_arg, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tmp_dir = os.path.join(output_dir, 'tmp')
    try:
        shutil.rmtree(tmp_dir)
    except:
        pass
    os.makedirs(tmp_dir, exist_ok=True)

    if test_arg.rank == 0:
        loader = tqdm(loader, ncols=20)

    for i, data in enumerate(loader):
        imgs, img_metas = data[:2]
        imgs = imgs.cuda()
        with torch.no_grad():
            tubes, _, _ = zip(*model(imgs, img_metas, return_loss=False))

        for img, tube, img_meta in zip(imgs, tubes, img_metas):
            # ===========================================VIS OUTPUT====================================================
            # if img is not None:
            #     vis_output(img.cpu(), img_meta, bbox.cpu(), stride=model_arg.frame_stride, out_folder='/home/pb/results/')
            # =========================================================================================================
            tube[:, [0, 5, 10]] += img_meta['start_frame']

            os.makedirs(os.path.join(tmp_dir, img_meta['video_name']), exist_ok=True)

            tube = tube.cpu().data.numpy()
            pickle.dump(tube, open(os.path.join(tmp_dir, img_meta['video_name'], str(img_meta['start_frame'])), 'wb'))

    synchronize()
    if test_arg.rank == 0:
        print('Finish prediction, Start matching')
        video_names = os.listdir(tmp_dir)
        pool = multiprocessing.Pool(processes=20)
        pool_list = []
        for vid in video_names:
            pool_list.append(pool.apply_async(match_video, (vid, tmp_dir, os.path.join(output_dir, 'res'), model_arg,)))
        for p in tqdm(pool_list, ncols=20):
            p.get()
        pool.close()
        pool.join()
        shutil.rmtree(tmp_dir)

        if test_arg.trainOrTest == 'train' and test_arg.dataset == 'MOT17':
            print("FINISH MATCHING, START EVALUATE")
            seq_map = 'MOT17_train.txt'
            evaluateTracking(seq_map, os.path.join(output_dir, 'res'),
                             os.path.join(test_arg.data_url, 'train'), 'MOT17')
        # elif test_arg.trainOrTest == 'train' and test_arg.dataset == 'MOT15':
        #     print("FINISH MATCHING, START EVALUATE")
        #     seq_map = 'MOT15_train.txt'
        #     evaluateTracking(seq_map, os.path.join(output_dir, 'res'),
        #                      os.path.join(test_arg.data_url[3], 'train'), 'MOT15')


def main(test_arg, model_arg):
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    local_rank = int(os.environ["LOCAL_RANK"])
    print('Rank: ' + str(test_arg.rank) + " Start!")
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print("Building TubeTK Model")

    model = TubeTK(num_classes=1, arg=model_arg, pretrained=False)

    data_loader = Data_Loader_MOT(
        batch_size=test_arg.batch_size,
        num_workers=8,
        input_path=test_arg.data_url,
        train_epoch=1,
        test_epoch=1,
        model_arg=model_arg,
        dataset=test_arg.dataset,
        test_seq=None,
        test_type=test_arg.trainOrTest,
    )

    model = model.cuda(local_rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if test_arg.apex:
        model = amp.initialize(model, opt_level='O1')

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    if test_arg.local_rank == 0:
        print("Loading Model")
    checkpoint = torch.load(test_arg.model_path + '/' + test_arg.model_name, map_location=
                            {'cuda:0': 'cuda:' + str(test_arg.local_rank),
                             'cuda:1': 'cuda:' + str(test_arg.local_rank),
                             'cuda:2': 'cuda:' + str(test_arg.local_rank),
                             'cuda:3': 'cuda:' + str(test_arg.local_rank),
                             'cuda:4': 'cuda:' + str(test_arg.local_rank),
                             'cuda:5': 'cuda:' + str(test_arg.local_rank),
                             'cuda:6': 'cuda:' + str(test_arg.local_rank),
                             'cuda:7': 'cuda:' + str(test_arg.local_rank)})
    model.load_state_dict(checkpoint['state'], strict=False)
    if test_arg.local_rank == 0:
        print("Finish Loading")
    del checkpoint

    model.eval()
    loader = data_loader.test_loader

    evaluate(model, loader, test_arg, model_arg, output_dir=test_arg.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--model_path', default='./models', type=str, help='model path')
    parser.add_argument('--model_name', default='TubeTK', type=str, help='model name')
    parser.add_argument('--data_url', default='./data/', type=str, help='model path')
    parser.add_argument('--output_dir', default='./link_res', type=str, help='output path')
    parser.add_argument('--apex', action='store_true', help='whether use apex')
    parser.add_argument('--config', default='./configs/TubeTK_resnet_50_FPN_8frame_1stride.yaml', type=str, help='config file')
    parser.add_argument('--dataset', default='MOT17', type=str, help='test which dataset: MOT17, MOT15')
    parser.add_argument('--trainOrTest', default='test', type=str, help='evaluate train or test set')

    parser.add_argument('--local_rank', type=int, help='gpus')

    test_arg, unparsed = parser.parse_known_args()

    model_arg = __C
    if test_arg.config is not None:
        cfg_from_file(test_arg.config)

    test_arg.rank = int(os.environ["RANK"])

    main(test_arg, model_arg)

