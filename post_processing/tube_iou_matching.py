import torch
import cv2
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from network.utils import bbox_iou
from datetime import datetime
import multiprocessing
from scipy.optimize import linear_sum_assignment


class Track:
    '''
    Track is the class of track. it contains all the node and manages the node. it contains the following information:
    1) all the nodes
    2) track id. it is unique it identify each track
    3) track pool id. it is a number to give a new id to a new track
    4) age. age indicates how old is the track
    5) max_age. indicates the dead age of this track
    '''
    _id_pool = 1
    def __init__(self):
        self.nodes = list()
        self.frames = {}
        self.mid_frames = {}
        self.id = Track._id_pool
        Track._id_pool += 1
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())
        self.prev_direction = None

    def update_frames(self, all_tube_frames, tube_boxes, mid_frame, mid_box, score, tube_direction):

        for frame, frame_box in zip(all_tube_frames, tube_boxes):
            if frame not in self.frames:
                self.frames[frame] = [frame_box, 1, score]
            else:
                self.frames[frame][0] += frame_box.astype(np.float)
                self.frames[frame][1] += 1
                self.frames[frame][2] += score

        if mid_frame not in self.mid_frames:
            self.mid_frames[mid_frame] = [mid_box.astype(np.float), 1, score]
        else:
            self.mid_frames[mid_frame][0] += mid_box.astype(np.float)
            self.mid_frames[mid_frame][1] += 1
            self.mid_frames[mid_frame][2] += score

        def get_center(box):
            return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        
        front_frame = np.max(all_tube_frames)
        back_frame = np.min(all_tube_frames)

        end_box = self.frames[front_frame][0] / self.frames[front_frame][1]
        start_box = self.frames[back_frame][0] / self.frames[back_frame][1]
        self.prev_direction = np.zeros(3)
        self.prev_direction[:2] = get_center(end_box) - get_center(start_box)
        self.prev_direction[2] = front_frame - back_frame


def track_tube_iou(track_boxes, tube_boxes):
    track_boxes = np.atleast_3d(track_boxes).astype(np.float)  # (n_track, n_tbbox, 4)
    tube_boxes = np.atleast_2d(tube_boxes).astype(np.float)    # (n_tbbox, 4)

    def track_tube_overlaps(bboxes1, bboxes2):
        lt = np.maximum(np.minimum(bboxes1[:, :, :2], bboxes1[:, :, 2:]), np.minimum(bboxes2[:, :2], bboxes2[:, 2:]))  # [rows, 2]
        rb = np.minimum(np.maximum(bboxes1[:, :, 2:], bboxes1[:, :, :2]), np.maximum(bboxes2[:, 2:], bboxes2[:, :2]))  # [rows, 2]
        wh = np.clip(rb - lt, 0, None)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        return overlap
    
    overlap = track_tube_overlaps(track_boxes, tube_boxes)

    area1 = (track_boxes[:, :, 2] - track_boxes[:, :, 0]) * (track_boxes[:, :, 3] - track_boxes[:, :, 1])
    area1 = np.abs(area1)
    area2 = (tube_boxes[:, 2] - tube_boxes[:, 0]) * (tube_boxes[:, 3] - tube_boxes[:, 1])
    area2 = np.abs(area2)

    ious = overlap / (area1 + area2 - overlap)

    return ious


def get_shape_diff(track_boxes, tube_boxes):
    track_boxes = np.atleast_3d(track_boxes).astype(np.float)  # (n_track, n_tbbox, 4)
    tube_boxes = np.atleast_2d(tube_boxes).astype(np.float)  # (n_tbbox, 4)

    track_height = track_boxes[:, :, 2] - track_boxes[:, :, 0]
    track_width = track_boxes[:, :, 3] - track_boxes[:, :, 1]
    tube_height = tube_boxes[:, 2] - tube_boxes[:, 0]
    tube_width = tube_boxes[:, 3] - tube_boxes[:, 1]

    diff = np.abs(track_height - tube_height) / (track_height + tube_height) + \
        np.abs(track_width - tube_width) / (track_width + tube_width)
    
    return np.exp(1.5 * -diff)


def update_tracks_fast(tracks, tube, arg):
    mid_frame = tube[0].astype(np.int)
    mid_box = tube[1:5]
    end_frame = tube[5].astype(np.int)
    end_box = tube[6:10]
    start_frame = tube[10].astype(np.int)
    start_box = tube[11:15]
    score = tube[15]

    def get_center(box):
        return np.array(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))

    back_frames = np.arange(start_frame, mid_frame)
    front_frames = np.arange(mid_frame + 1, end_frame + 1)
    all_tube_frames = np.arange(start_frame, end_frame + 1)
    
    back_start_coef = (mid_frame - back_frames) / (mid_frame - start_frame)
    back_mid_coef = (back_frames - start_frame) / (mid_frame - start_frame)
    front_mid_coef = (end_frame - front_frames) / (end_frame - mid_frame)
    front_end_coef = (front_frames - mid_frame) / (end_frame - mid_frame)

    back_frame_boxes = np.outer(back_start_coef, start_box) + np.outer(back_mid_coef, mid_box)
    front_frame_boxes = np.outer(front_end_coef, end_box) + np.outer(front_mid_coef, mid_box)

    tube_boxes = np.concatenate((back_frame_boxes, mid_box[None], front_frame_boxes))
    tube_frame_num = len(all_tube_frames)

    depth_divider = 8

    tube_direction = np.zeros(3)
    tube_direction[:2] = get_center(end_box) - get_center(start_box)
    tube_direction[2] = np.max(all_tube_frames) - np.min(all_tube_frames)
    tube_direction[2] /= depth_divider

    if len(tracks) == 0:
        new_track = Track()
        new_track.update_frames(all_tube_frames, tube_boxes, mid_frame, mid_box, score, tube_direction)
        tracks.append(new_track)
        return

    all_has_frame = np.zeros((len(tracks), tube_frame_num), dtype=np.bool)
    all_track_boxes = np.zeros((len(tracks), *tube_boxes.shape))
    track_direction = np.zeros((len(tracks), 3))

    for track_idx, track in enumerate(tracks):
        # overlap_area = [1e8, -1]
        if track.prev_direction is not None:
            track_direction[track_idx, :] = track.prev_direction

        for i, frame in enumerate(all_tube_frames):
            if frame not in track.frames:
                continue
            all_has_frame[track_idx, i] = True
            all_track_boxes[track_idx, i, :] = \
                track.frames[frame][0] / track.frames[frame][1]
            # overlap_area[0] = min(overlap_area[0], frame)
            # overlap_area[1] = max(overlap_area[1], frame)

        # if overlap_area[1] < 0:
        #     continue
        # while overlap_area[0] - 1 in track.frames and overlap_area[1] - overlap_area[0] + 1 < tube_frame_num:
        #     overlap_area[0] -= 1
        # while overlap_area[1] + 1 in track.frames and overlap_area[1] - overlap_area[0] + 1 < tube_frame_num:
        #     overlap_area[1] += 1
        # track_direction[track_idx, :2] = get_center(track.frames[overlap_area[1]][0] / track.frames[overlap_area[1]][1]) - \
        #         get_center(track.frames[overlap_area[0]][0] / track.frames[overlap_area[0]][1])
        # track_direction[track_idx, 2] = overlap_area[1] - overlap_area[0]

    track_direction[:, 2] /= depth_divider

    has_overlap = (np.sum(all_has_frame, axis=1) > 0)
    all_iou = np.zeros(all_has_frame.shape, dtype=np.float)
    shape_diff = np.zeros(all_has_frame.shape, dtype=np.float)
    all_iou[has_overlap] = track_tube_iou(all_track_boxes[has_overlap], tube_boxes)
    shape_diff[has_overlap] = get_shape_diff(all_track_boxes[has_overlap], tube_boxes)

    mean_all_iou = np.zeros(has_overlap.shape, dtype=np.float)
    mean_all_iou[has_overlap] = np.sum(all_iou[has_overlap], axis=1) / np.sum(all_has_frame[has_overlap], axis=1)

    angle_cos = np.ones_like(mean_all_iou)
    norm_mul = np.linalg.norm(track_direction, axis=1) * np.linalg.norm(tube_direction)

    cos_mask = np.logical_and(has_overlap, norm_mul > 0)
    angle_cos[cos_mask] = np.dot(track_direction[cos_mask], tube_direction) / norm_mul[cos_mask]

    mean_all_iou = mean_all_iou * (1 + arg.cos_weight * angle_cos)
    max_idx = np.argmax(mean_all_iou)

    if mean_all_iou[max_idx] > arg.linking_min_iou:
        tracks[max_idx].update_frames(all_tube_frames, tube_boxes, mid_frame, mid_box, score, tube_direction)
    else:
        new_track = Track()
        new_track.update_frames(all_tube_frames, tube_boxes, mid_frame, mid_box, score, tube_direction)
        tracks.append(new_track)


def filt_bbox(save_path):
    def bboxfilt(res, l=8):
        max_frame = np.max(res[0])
        range_mask = (res[6] >= l) | (res[0] <= 8) | (res[0] + 8 >= max_frame)
        return res[range_mask]

    def trackfilt(track, l=16):
        max_fid = int(np.max(track[0]))
        min_fid = int(np.min(track[0]))
        return max_fid - min_fid < 5
        # max_frame = np.max(track.iloc[:, 0])
        # range_mask = (track[0] > 8) & (track[0] + 8 < max_frame)
        # if np.mean(track[range_mask][6]) < l:
        #     return True
        # else:
        #     return False

    def ip_linear(det1, det2, fid):
        fid1 = det1[0]
        fid2 = det2[0]
        w1 = 1.0 * (fid2 - fid) / (fid2 - fid1)
        w2 = 1.0 * (fid - fid1) / (fid2 - fid1)

        ip = np.copy(det1)
        ip[0] = fid
        ip[2:6] = w1 * det1[2:6] + w2 * det2[2:6]
        return np.array([ip])

    def track_complete(track, gap_threshold=8):
        max_fid = int(np.max(track[:, 0]))
        min_fid = int(np.min(track[:, 0]))

        ips = []
        ip_cnt = 0
        max_missing_len = 0
        for i, fid in enumerate(list(track[:-1, 0])):
            if track[i+1, 0] - 1 != track[i, 0]:
                if track[i+1, 0] - track[i, 0] - 1 > gap_threshold:
                    continue
                cur_fid = track[i, 0] + 1
                missing_len = 0
                while cur_fid < track[i+1, 0]:
                    ips.append(ip_linear(track[i+1], track[i], cur_fid))
                    cur_fid = cur_fid + 1
                    missing_len = missing_len + 1
                ip_cnt = ip_cnt + missing_len
                max_missing_len = max(max_missing_len, missing_len)
        
        assert len(ips) == ip_cnt, (track, ips)
        ips.append(track)
        new_track = np.concatenate(ips, axis=0)
        new_track = new_track[new_track[:, 0].argsort()]
        if ip_cnt == 0:
            return track, 0
        else:
            return new_track, ip_cnt

    param_pairs = [
        (['05'], [0, 4, 8]),
        (['10'], [0, 6, 8]),
        (['11'], [0, 6, 8]),
        (['13'], [0, 9, 8]),
        (['02'], [0, 6, 8]),
        (['09'], [0, 4, 8]),
        (['04'], [0, 12, 8]),
        (['06'], [0, 4, 8]),
        (['07'], [0, 6, 8]),
        (['12'], [0, 6, 8]),
        (['14'], [0, 9, 8]),
        (['01'], [0, 6, 30]),
        (['08'], [0, 4, 30]),
        (['03'], [0, 12, 30])
    ]
    params = {}
    for file_nums, param in param_pairs:
        params.update({x: param for x in file_nums})
    file_num = None
    for k in params.keys():
        if k in save_path and file_num is None:
            file_num = k
        elif k in save_path:
            assert False
    assert file_num is not None
    res = pd.read_csv(save_path, header=None)

    min_num = params[file_num][0]
    min_bbox = params[file_num][1]
    res = bboxfilt(res, min_bbox)
    filtered_tracks = [x[0] for x in res.groupby(1) if trackfilt(x[1], min_num)]
    inds = [res.iloc[x, 1] not in filtered_tracks for x in range(len(res))]
    res = res[inds]
    inds = np.unique(res[1])
    dict_map = {x: i + 1 for i, x in enumerate(inds)}
    res[1] = res[1].map(lambda x: dict_map[x])
#    res.to_csv(save_path, header=None, index=False)

    # track complete part
    tracks = res.groupby(1)
    new_tracks = []
    for tid in tracks.groups.keys():
        res, _ = track_complete(tracks.get_group(tid).values, params[file_num][2])
        if res is not None:
            new_tracks.append(res)

    new_tracks = np.concatenate(new_tracks)
    new_tracks = new_tracks[new_tracks[:, 0].argsort()]
    np.savetxt(save_path, new_tracks, fmt='%i,%i,%f,%f,%f,%f,%i,%f,%i,%i', delimiter=',')


def final_processing(tracks, save_path, mid_only):
    res = []
    assert len(tracks) != 0, 'No Tracks: ' + str(save_path)
    for track in tracks:
        if mid_only:
            frames = track.mid_frames
        else:
            frames = track.frames
        cur_res = np.zeros((len(track.mid_frames), 10))
        for i, (frame, bbox) in enumerate(track.mid_frames.items()):
            cur_res[i, 0] = frame + 1
            cur_res[i, 2:6] = bbox[0] / bbox[1]
            cur_res[i, 6] = track.frames[frame][1]  # num of average bbox, use all frames
            cur_res[i, 7] = track.frames[frame][2] / track.frames[frame][1]  # average score, use all frames
        cur_res[:, 1] = track.id
        res.append(cur_res)
    res = np.concatenate(res)
    res = res[res[:, 0].argsort()]
    res[:, -2:] = -1
    res[:, 4:6] -= res[:, 2:4]
    if save_path is not None:
        try:
            if save_path[0] == '/':
                os.makedirs(os.path.join('/', *(save_path.split('/')[:-1])))
            else:
                os.makedirs(os.path.join(*(save_path.split('/')[:-1])))
        except:
            pass
        np.savetxt(save_path, res, fmt='%i,%i,%f,%f,%f,%f,%i,%f,%i,%i', delimiter=',')
        filt_bbox(save_path)
    # ? return res or track


def archive_tracks(tracks, arch_tracks, cur_frame, forward_frames):
    track_ = []
    for track in tracks:
        max_frame = max(track.frames.keys())
        if max_frame + 2 * forward_frames < cur_frame:
            arch_tracks.append(track)
        else:
            track_.append(track)

    return track_


def adjust_poi_tubes(tubes, poi_tubes):
    def adjust_single_frame(tubes, poi_tubes):
        tubes_end_mask = tubes[:, 5] > tubes[:, 0]
        # Trans from end to mid
        trans_x_end = (tubes[tubes_end_mask][:, 6] + tubes[tubes_end_mask][:, 8]) / 2 - \
            (tubes[tubes_end_mask][:, 1] + tubes[tubes_end_mask][:, 3]) / 2
        trans_y_end = (tubes[tubes_end_mask][:, 7] + tubes[tubes_end_mask][:, 9]) / 2 - \
            (tubes[tubes_end_mask][:, 2] + tubes[tubes_end_mask][:, 4]) / 2
        # Trans Per Frame
        trans_x_end = trans_x_end / (tubes[tubes_end_mask][:, 5] - tubes[tubes_end_mask][:, 0])
        trans_y_end = trans_y_end / (tubes[tubes_end_mask][:, 5] - tubes[tubes_end_mask][:, 0])
        # Trans Per Height
        mean_trans_x_end = np.mean(trans_x_end / (tubes[tubes_end_mask][:, 7] - tubes[tubes_end_mask][:, 9]))
        mean_trans_y_end = np.mean(trans_y_end / (tubes[tubes_end_mask][:, 7] - tubes[tubes_end_mask][:, 9]))
        poi_tubes[:, [6, 8]] += (mean_trans_x_end * (poi_tubes[:, 5] - poi_tubes[:, 0])
                                 * (poi_tubes[:, 7] - poi_tubes[:, 9]))[:, None]
        poi_tubes[:, [7, 9]] += (mean_trans_y_end * (poi_tubes[:, 5] - poi_tubes[:, 0])
                                 * (poi_tubes[:, 7] - poi_tubes[:, 9]))[:, None]

        tubes_start_mask = tubes[:, 10] < tubes[:, 0]
        trans_x_start = (tubes[tubes_start_mask][:, 11] + tubes[tubes_start_mask][:, 13]) / 2 - \
            (tubes[tubes_start_mask][:, 1] + tubes[tubes_start_mask][:, 3]) / 2
        trans_y_start = (tubes[tubes_start_mask][:, 12] + tubes[tubes_start_mask][:, 14]) / 2 - \
            (tubes[tubes_start_mask][:, 2] + tubes[tubes_start_mask][:, 4]) / 2
        # Trans Per Frame
        trans_x_start = trans_x_start / (tubes[tubes_start_mask][:, 10] - tubes[tubes_start_mask][:, 0])
        trans_y_start = trans_y_start / (tubes[tubes_start_mask][:, 10] - tubes[tubes_start_mask][:, 0])
        # Trans Per Height
        mean_trans_x_start = np.mean(trans_x_start / (tubes[tubes_start_mask][:, 12] - tubes[tubes_start_mask][:, 14]))
        mean_trans_y_start = np.mean(trans_y_start / (tubes[tubes_start_mask][:, 12] - tubes[tubes_start_mask][:, 14]))
        poi_tubes[:, [11, 13]] += (mean_trans_x_start * (poi_tubes[:, 10] - poi_tubes[:, 0])
                                   * (poi_tubes[:, 12] - poi_tubes[:, 14]))[:, None]
        poi_tubes[:, [12, 14]] += (mean_trans_y_start * (poi_tubes[:, 10] - poi_tubes[:, 0])
                                   * (poi_tubes[:, 12] - poi_tubes[:, 14]))[:, None]

        return poi_tubes

    frame_idxs = np.unique(tubes[:, 0])
    for frame_idx in frame_idxs:
        poi_tubes[poi_tubes[:, 0] == frame_idx] = adjust_single_frame(
            tubes[tubes[:, 0] == frame_idx], poi_tubes[poi_tubes[:, 0] == frame_idx])
    
    return poi_tubes


def matching(tubes, arg, save_path=None, verbose=False, mid_only=True, poi_tubes=None):
    """
    tubes: All tubes in a video to match. (n, 15 + 1) [mid_frame, mid_box, front_frame, front_box, back_frame, back_box, value]
    save_path: File path to save formatted result.
    """
    tracks = []
    if not isinstance(tubes, np.ndarray):
        tubes = tubes.cpu().data.numpy()
    
    if poi_tubes is not None:
        poi_tubes = adjust_poi_tubes(tubes, poi_tubes)
        tubes = np.concatenate((tubes, poi_tubes))
    
    tubes = tubes[(-tubes[:, 15]).argsort()]
    tubes = tubes[tubes[:, 0].argsort(kind='stable')]
    arch_tracks = []
    prev_frame = -1
    tubes_one_frame = 0

    for tube in tubes:
        update_tracks_fast(tracks, tube, arg)

        current_frame = tube[0]
        if prev_frame != current_frame and prev_frame != -1:  # Switch Frame
            if verbose:
                print('{}\tFrame: {}\tTubes: {}\tCur tracks:{}\tArch tracks:{}'.format(
                    datetime.now().time(), prev_frame, tubes_one_frame, len(tracks), len(arch_tracks)))
            tubes_one_frame = 0
            # Archive tracks 2*forward_frames frames away, they won't be useful anymore
            if int(current_frame) % 10 == 0:
                tracks = archive_tracks(tracks, arch_tracks, current_frame, arg.forward_frames * arg.frame_stride)

        prev_frame = current_frame
        tubes_one_frame += 1

    arch_tracks.extend(tracks)
    tracks = arch_tracks
    final_processing(tracks, save_path, mid_only)
    return tracks
