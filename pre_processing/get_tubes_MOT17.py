import numpy as np
import os
import pandas as pd
from network.utils import bbox_iou
import pickle
from tqdm import tqdm
import shutil
import multiprocessing
from configs.default import __C, cfg_from_file
from dataset.Parsers.structures import *
import argparse


class GTSingleParser:
    def __init__(self, folder,
                 min_visibility,
                 forward_frames,
                 frame_stride,
                 tube_thre):
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder
        self.forward_frames = forward_frames
        self.tube_thre = tube_thre
        self.min_visibility = min_visibility
        self.frame_stride = frame_stride

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]  # human class
        gt_file = gt_file[gt_file[8] > min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)
        # 3. update tracks
        self.tracks = Tracks()
        self.recorder = {}
        for key in gt_group_keys:
            det = gt_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            det = np.array(det[:, 2:6])
            det[:, 2:4] += det[:, :2]

            self.recorder[key - 1] = list()
            # 3.1 update tracks
            for id, d in zip(ids, det):
                node = Node(d, key - 1)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key - 1].append((track_index, node_index))

    def bbox2tube(self, track, mid_id, direction, pos_in_video, thre):
        def get_true_z(mid_node, end_node):
            return end_node.frame_id - mid_node.frame_id

        def get_inter_box(start_box, end_box, inter_id, end_id):
            return start_box * (end_id - inter_id) / end_id + end_box * inter_id / end_id

        mid_node = track.get_node_by_index(mid_id)
        mid_box = mid_node.box
        inter_boxes = []

        z = 1 if direction == 'front' else -1
        if mid_id + z >= len(track.nodes) or mid_id + z < 0:
            return np.array([0, 0, 0, 0, 0])

        true_z = get_true_z(mid_node, track.get_node_by_index(mid_id + z))

        max_len = (self.forward_frames * 2 - 1) * self.frame_stride + 1

        while -1 * pos_in_video <= true_z < max_len - pos_in_video:
            iou_total = 0
            end_node = track.get_node_by_index(mid_id + z)
            end_box = end_node.box
            for i, gt_box in enumerate(inter_boxes):
                iou = sum(bbox_iou(gt_box[None], get_inter_box(mid_box, end_box, i + 1, len(inter_boxes) + 1)[None]))
                iou_total += iou
            iou_total += 1
            iou_total /= (len(inter_boxes) + 1)

            if iou_total < thre:
                break

            inter_boxes.append(end_box)
            if z % self.frame_stride == 0:
                res_z = true_z

            z += 1 if direction == 'front' else -1
            if mid_id + z >= len(track.nodes) or mid_id + z < 0:
                break
            true_z = get_true_z(mid_node, track.get_node_by_index(mid_id + z))

        if not inter_boxes or len(inter_boxes) < self.frame_stride:
            return np.array([0, 0, 0, 0, 0])
        else:
            ret_ind = (len(inter_boxes) // self.frame_stride) * self.frame_stride - 1
            return np.concatenate((np.array([abs(res_z)]), inter_boxes[ret_ind] - mid_box))

    def get_item(self, frame_index):
        start_frame = frame_index
        max_len = (self.forward_frames * 2 - 1) * self.frame_stride + 1
        if self.max_frame_index - start_frame < max_len:
            return 0
        # if not frame_index in self.recorder:
        #     return 0

        tubes = []
        for i in range(self.forward_frames * 2):
            frame_index = start_frame + i * self.frame_stride
            if frame_index not in self.recorder:
                continue

            det_ids = self.recorder[frame_index]

            # 1. get tubes
            for track_index, node_index in det_ids:
                t = self.tracks.get_track_by_index(track_index)
                n = t.get_node_by_index(node_index)
                mid_box = np.concatenate((n.box, np.array([frame_index - start_frame])))
                # backward
                back_box = self.bbox2tube(track=t, mid_id=node_index, direction='back',
                                          pos_in_video=i * self.frame_stride, thre=self.tube_thre)
                # forward
                front_box = self.bbox2tube(track=t, mid_id=node_index, direction='front',
                                           pos_in_video=i * self.frame_stride, thre=self.tube_thre)
                tube = np.concatenate((mid_box, front_box, back_box))
                tubes.append(tube)

        if len(tubes) == 0:
            return 0
        tubes = np.array(tubes)
        try:
            os.makedirs(os.path.join(self.folder, 'tubes_' + str(self.forward_frames) + '_' + str(self.frame_stride) + '_' + str(self.min_visibility)))
        except:
            pass
        pickle.dump(tubes, open(os.path.join(self.folder, 'tubes_' + str(self.forward_frames) + '_' + str(self.frame_stride) + '_' + str(self.min_visibility), str(start_frame)), 'wb'))
        return 0

    def clear(self):
        try:
            shutil.rmtree(os.path.join(self.folder, 'tubes_' + str(self.forward_frames) + '_' + str(self.frame_stride) + '_' + str(self.min_visibility)))
        except:
            pass

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, mot_root,
                 arg,
                 type='train',
                 ):
        # analsis all the folder in mot_root
        # 1. get all the folders
        mot_root = os.path.join(mot_root, type)
        all_folders = sorted(
            [os.path.join(mot_root, i) for i in os.listdir(mot_root)
             if os.path.isdir(os.path.join(mot_root, i))
             and i.find('FRCNN') != -1]
        )
        # 2. create single parser
        self.parsers = [GTSingleParser(folder, forward_frames=arg.forward_frames,
                                       min_visibility=arg.min_visibility,
                                       frame_stride=arg.frame_stride,
                                       tube_thre=arg.tube_thre) for folder in all_folders]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def clear(self):
        print('Clearing')
        for parser in tqdm(self.parsers, ncols=20):
            parser.clear()

    def run(self):
        print('Running')
        pool = multiprocessing.Pool(processes=40)
        pool_list = []
        for item in tqdm(range(self.len), ncols=20):
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

            if index >= len(self.parsers):
                return
            pool_list.append(pool.apply_async(self.parsers[index].get_item, (current_item,)))
            # self.parsers[index].get_item(current_item)
        for p in tqdm(pool_list, ncols=20):
            p.get()
        pool.close()
        pool.join()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mot_root', default='./data', type=str, help="mot data root")
    arg, unparsed = arg_parser.parse_known_args()
    config = __C
    cfg_from_file('../configs/get_MOT17_tube.yaml')
    parser = GTParser(mot_root=arg.mot_root, arg=config)
    parser.clear()
    parser.run()
