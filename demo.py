import pickle
import os
import numpy as np
import torch
import warnings
from tqdm import tqdm
from network.tubetk import TubeTK
from apex import amp
import argparse
import multiprocessing
from configs.default import __C, cfg_from_file
from post_processing.tube_iou_matching import matching
warnings.filterwarnings('ignore')
import shutil
from Visualization.Vis_Res import vis_one_video
import cv2
import torch.utils.data as data
import random
from dataset.augmentation import SSJAugmentation


class GTSingleParser:
    def __init__(self, video,
                 forward_frames=4,
                 frame_stride=1,
                 min_vis=-0.1,
                 value_range=1):
        self.frame_stride = frame_stride
        self.value_range = value_range
        self.video_name = video
        self.min_vis = min_vis
        self.forward_frames = forward_frames

        self.cap = cv2.VideoCapture(video)
        fps = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_counter = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_frames = np.zeros((frame_counter, height, width, 3), dtype='float16')
        cnt = 0
        if int(os.environ["RANK"]) == 0:
            print('reading video...')
            pbar = tqdm(total=frame_counter)

        os.makedirs(video + '_imgs', exist_ok=True)
        while self.cap.isOpened():
            _, frame = self.cap.read()
            if cnt >= frame_counter:
                break
            if frame is not None:
                frame_ok = frame  # .astype('float16')
            else:
                if int(os.environ['RANK']) == 0:
                    print('cannot read frame')
            self.video_frames[cnt] = frame_ok
            cv2.imwrite(filename=os.path.join(video + '_imgs', str(cnt + 1) + '.jpg'), img=frame_ok)  #.astype('int8'))
            cnt += 1
            if int(os.environ["RANK"]) == 0:
                pbar.update(1)
        if int(os.environ["RANK"]) == 0:
            print('finish_reading')
            pbar.close()

        self.max_frame_index = frame_counter - (
                    self.forward_frames * 2 - 1) * self.frame_stride

    def _getimage(self, frame_index):
        img = self.video_frames[frame_index]
        return img

    def get_item(self, frame_index):
        frame_stride = self.frame_stride

        start_frame = frame_index
        max_len = self.forward_frames * 2 * frame_stride

        # get image meta
        img_meta = {}
        image = self._getimage(frame_index)
        img_meta['img_shape'] = [max_len, image.shape[0], image.shape[1]]
        img_meta['value_range'] = self.value_range
        img_meta['pad_percent'] = [1, 1]  # prepared for padding
        img_meta['video_name'] = os.path.basename(self.video_name)
        img_meta['start_frame'] = start_frame

        # get image
        imgs = []
        for i in range(self.forward_frames * 2):
            frame_index = start_frame + i * frame_stride
            image = self._getimage(frame_index)  # h, w, c
            imgs.append(image)

        # get_tube
        tubes = np.zeros((1, 15))
        num_dets = len(tubes)
        labels = np.ones((num_dets, 1))  # only human class

        tubes = np.array(tubes)
        imgs = np.array(imgs)

        return imgs, img_meta, tubes, labels, start_frame

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, data_root,
                 forward_frames=4,
                 frame_stride=1,
                 min_vis=-0.1,
                 value_range=1):
        # analsis all the folder in mot_root
        # 1. get all the folders
        all_videos = sorted([os.path.join(data_root, i) for i in os.listdir(data_root)
                             if '_imgs' not in i])
        # 2. create single parser
        self.parsers = [GTSingleParser(video, forward_frames=forward_frames, frame_stride=frame_stride,
                                       min_vis=min_vis, value_range=value_range) for video in all_videos]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        # 1. find the parser
        total_len = 0
        index = 0
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None, None, None, None
        return self.parsers[index].get_item(current_item)


class DemoDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''
    def __init__(self,
                 data_root,
                 arg,
                 transform=SSJAugmentation,
                 ):
        # 1. init all the variables
        self.data_root = data_root
        self.transform = transform(size=arg.img_size, type='test')

        # 2. init GTParser
        self.parser = GTParser(self.data_root, forward_frames=arg.forward_frames,
                               frame_stride=arg.frame_stride, min_vis=arg.min_visibility,
                               value_range=arg.value_range)

    def __getitem__(self, item):
        item = item % len(self.parser)
        image, img_meta, tubes, labels, start_frame = self.parser[item]

        while image is None:
            image, img_meta, tubes, labels, start_frame = self.parser[(item+random.randint(-10, 10)) % len(self.parser)]
            print('None processing.')

        if self.transform is None:
            return image, img_meta, tubes, labels, start_frame
        else:
            image, img_meta, tubes, labels, start_frame = self.transform(image, img_meta, tubes, labels, start_frame)
            return image, img_meta, tubes, labels, start_frame

    def __len__(self):
        return len(self.parser)


class Data_Loader():
    def __init__(self,
                 batch_size,
                 num_workers,
                 input_path,
                 model_arg):
        self.num_workers = num_workers
        self.BATCH_SIZE = batch_size

        def my_collate(batch):
            imgs = torch.stack([torch.tensor(item[0]) for item in batch], 0)
            img_meta = [item[1] for item in batch]
            tubes = [item[2] for item in batch]
            labels = [item[3] for item in batch]
            start_frame = [item[4] for item in batch]
            return imgs, img_meta, tubes, labels, start_frame

        self.demo_set = DemoDataset(data_root=input_path, arg=model_arg)

        if int(os.environ["RANK"]) == 0:
            print('==> Validation data :', len(self.demo_set))
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.demo_set)
        self.loader = torch.utils.data.DataLoader(
            dataset=self.demo_set,
            batch_size=self.BATCH_SIZE,
            collate_fn=my_collate,
            num_workers=self.num_workers,
            pin_memory=True, sampler=val_sampler)


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

        print('Finish matching, Start writing to video')
        for vid in os.listdir(os.path.join(output_dir, 'res')):
            cap = cv2.VideoCapture(os.path.join(test_arg.video_url, vid[0: -4]))
            frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            res_file = os.path.join(output_dir, 'res', vid)
            img_dir = os.path.join(test_arg.video_url, vid[0: -4] + '_imgs')
            output_name = os.path.join(test_arg.output_dir, vid + '.avi')
            vis_one_video(res_file, frame_rate, img_width, img_height, img_dir, output_name)
            try:
                shutil.rmtree(img_dir)
            except:
                pass


def main(test_arg, model_arg):
    torch.distributed.init_process_group(backend="nccl", init_method='env://')

    local_rank = int(os.environ["LOCAL_RANK"])
    print('Rank: ' + str(test_arg.rank) + " Start!")
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print("Building TubeTK Model")

    model = TubeTK(num_classes=1, arg=model_arg, pretrained=False)

    data_loader = Data_Loader(
        batch_size=test_arg.batch_size,
        num_workers=8,
        input_path=test_arg.video_url,
        model_arg=model_arg,
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
    loader = data_loader.loader

    evaluate(model, loader, test_arg, model_arg, output_dir=test_arg.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--model_path', default='./models', type=str, help='model path')
    parser.add_argument('--model_name', default='TubeTK', type=str, help='model name')
    parser.add_argument('--video_url', type=str, default='./data', help='video path')
    parser.add_argument('--output_dir', default='./vis_video', type=str, help='output path')
    parser.add_argument('--apex', action='store_true', help='whether use apex')
    parser.add_argument('--config', default='./configs/TubeTK_resnet_50_FPN_8frame_1stride.yaml', type=str, help='config file')

    parser.add_argument('--local_rank', type=int, help='gpus')

    test_arg, unparsed = parser.parse_known_args()

    model_arg = __C
    if test_arg.config is not None:
        cfg_from_file(test_arg.config)

    test_arg.rank = int(os.environ["RANK"])

    main(test_arg, model_arg)

