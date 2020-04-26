import numpy as np
import os
import pandas as pd
from network.utils import bbox_iou
import pickle
from tqdm import tqdm
import argparse
import multiprocessing
from configs.default import __C, cfg_from_file
from dataset.Parsers.structures import *


class GTSingleParser:
    def __init__(self, folder,
                 min_visibility,
                 forward_frames,
                 frame_stride,
                 tube_thre,
                 loose,
                 height_clamp):
        # 1. get the gt path and image folder
        split_path = folder.split('/')
        if folder[0] == '/':
            jta_root = '/' + os.path.join(*split_path[:-3])
        else:
            jta_root = os.path.join(*split_path[:-3])
        type = split_path[-2]
        video_name = split_path[-1]
        gt_file_path = os.path.join(jta_root, 'gt_' + str(loose) + '_' + str(min_visibility) + '_' + str(height_clamp), type, video_name, 'gt.txt')
        # gt_file_path = os.path.join(folder, 'gt/gt.txt')

        self.folder = folder
        self.forward_frames = forward_frames
        self.tube_thre = tube_thre
        self.min_visibility = min_visibility
        self.frame_stride = frame_stride

        self.tube_res_path = os.path.join(jta_root,
                                          'tubes_' + str(self.forward_frames) + '_' + str(
                                              self.frame_stride) + '_' + str(self.min_visibility),
                                          type,
                                          video_name)

        try:
            os.makedirs(self.tube_res_path)
        except:
            pass

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

                # remove the fast turning
                mid_box_w = mid_box[2] - mid_box[0]
                if abs(front_box[0] - back_box[0]) > 2.5 * mid_box_w:
                    print('remove turning')
                    continue

                tube = np.concatenate((mid_box, front_box, back_box))
                tubes.append(tube)

        if len(tubes) == 0:
            return 0
        tubes = np.array(tubes)

        pickle.dump(tubes, open(os.path.join(self.tube_res_path, str(start_frame)), 'wb'))
        return 0

    def __len__(self):
        return self.max_frame_index


class GTParser:
    def __init__(self, jta_root,
                 arg,
                 loose,
                 height_clamp,
                 type='train',
                 ):
        # analsis all the folder in jta_root
        # 1. get all the folders
        self.jta_root = jta_root
        jta_root = os.path.join(jta_root, type)
        all_folders = sorted(
            [os.path.join(jta_root, i) for i in os.listdir(jta_root)
             if os.path.isdir(os.path.join(jta_root, i))]
        )
        # 2. create single parser
        print('Init SingleParser')
        self.parsers = [GTSingleParser(folder, forward_frames=arg.forward_frames,
                                       min_visibility=arg.min_visibility,
                                       frame_stride=arg.frame_stride,
                                       tube_thre=arg.tube_thre,
                                       loose=loose,
                                       height_clamp=height_clamp) for folder in tqdm(all_folders, ncols=20)]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers]
        self.len = sum(self.lens)

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def clear(self):
        print('Clearing')

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


def get_gt(json_path, frames_path, loose, min_visiblity, height_clamp):
    assert os.path.exists(json_path), 'File does not exist: {}'.format(json_path)
    assert os.path.exists(frames_path), 'Folder does not exist: {}'.format(frames_path)
    split_path = frames_path.split('/')
    if frames_path[0] == '/':
        jta_root = '/' + os.path.join(*split_path[:-3])
    else:
        jta_root = os.path.join(*split_path[:-3])
    type = split_path[-2]
    video_name = split_path[-1]
    gt_path = os.path.join(jta_root, 'gt_' + str(loose) + '_' + str(min_visiblity) + '_' + str(height_clamp), type, video_name)
    try:
        os.makedirs(gt_path)
    except:
        pass
    gt_file = os.path.join(gt_path, 'gt.txt')
    df = pd.read_json(json_path)
    df = df.iloc[:, [0, 1, 3, 4, 8]]  # Frame, ID, x, y, occluded
    df_group = df.groupby([0, 1])  # Group by frame and id

    def get_bbox(g):
        assert len(g.columns) == 5
        if g.iloc[:, 4].sum() >= (1 - min_visiblity) * len(g):  # Completely occluded
            return pd.Series([-1, 0, 0, 0, 0, 0, 0], dtype=np.int)
        x1 = np.maximum(0, g.iloc[:, 2].min())
        y1 = np.maximum(0, g.iloc[:, 3].min())
        x2 = np.minimum(1920, g.iloc[:, 2].max())
        y2 = np.minimum(1080, g.iloc[:, 3].max())
        w = x2 - x1
        h = y2 - y1
        # Loose a little bit
        x1 -= np.round(w * loose)
        y1 -= np.round(h * loose)
        x1 = np.maximum(0.0, x1)
        y1 = np.maximum(0.0, y1)
        w = np.round(w * (1 + loose*2))
        h = np.round(h * (1 + loose*2))
        w = np.minimum(1920 - x1, w)
        h = np.minimum(1080 - y1, h)

        return pd.Series([x1, y1, w, h, 1, 1, 1], dtype=np.int)

    res_df = df_group.apply(get_bbox)
    res_df = res_df[res_df.iloc[:, 0] != -1]

    # get mode and remove the small box
    ns, edges = np.histogram(res_df.iloc[:, 3], bins=50)
    max_n = np.argmax(ns)
    mode = np.mean(edges[[max_n, max_n + 1]])
    res_df = res_df[res_df.iloc[:, 3] > height_clamp * mode]
    res_df = res_df[res_df.iloc[:, 3] > 7]

    res_df.to_csv(gt_file, header=False)


def get_gts(jta_root, frames_dir, loose, min_vis, height_clamp):
    pool = multiprocessing.Pool(processes=20)
    pool_list = []
    anno_path = os.path.join(jta_root, 'annotations')
    for type in os.listdir(anno_path):
        for json_file in os.listdir(os.path.join(anno_path, type)):
            json_path = os.path.join(anno_path, type, json_file)
            frames_path = os.path.join(jta_root, frames_dir, type, os.path.splitext(json_file)[0])
            pool_list.append(pool.apply_async(get_gt, (json_path, frames_path, loose, min_vis, height_clamp, )))

    for p in tqdm(pool_list, ncols=20):
        p.get()
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jta_root', type=str, help="data path of jta")
    parser.add_argument('--loose', type=float, default=0.1, help="ratio to loose the bbox generated from keypoint")
    parser.add_argument('--height_clamp', type=float, default=0.6, help="get rid of the bboxes whose height is smaller "
                                                                        "than 0.6 of the mean height")
    arg_input, unparsed = parser.parse_known_args()

    arg = __C
    cfg_from_file('../configs/get_jta_tube.yaml')

    print('Generating GT files')
    get_gts(jta_root=arg_input.jta_root, frames_dir='imgs', loose=arg_input.loose, min_vis=arg.min_visibility,
            height_clamp=arg_input.height_clamp)
    print('Generating Tubes')
    parser = GTParser(jta_root=os.path.join(arg_input.jta_root, 'imgs'), arg=arg, type='train', loose=arg_input.loose,
                      height_clamp=arg_input.height_clamp)
    parser.clear()
    parser.run()
