import torch
import cv2
import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from network.utils import bbox_iou
from datetime import datetime


class Node:

    def __init__(self, box):
        self.box = box


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
    ''' for mot
    '''
    _max_num_node = 36
    '''for kitti
    _max_num_node = 5
    '''
    def __init__(self):
        self.nodes = list()
        self.frames = {}
        self.id = Track._id_pool
        Track._id_pool += 1
        self.color = tuple((np.random.rand(3) * 255).astype(int).tolist())

    def __del__(self):
        for n in self.nodes:
            del n

    def update_frames(self, node):
        tube = node.box

        mid_frame = tube[0].astype(np.int)
        mid_box = tube[1:5]
        end_frame = tube[5].astype(np.int)
        end_box = tube[6:10]
        start_frame = tube[10].astype(np.int)
        start_box = tube[11:15]
        score = tube[15]

        for frame in range(start_frame, mid_frame):
            frame_box = start_box * (mid_frame - frame) / (mid_frame - start_frame) + mid_box * (frame - start_frame) / (mid_frame - start_frame)
            if frame not in self.frames:
                self.frames[frame] = [frame_box, 1, score]
            else:
                self.frames[frame][0] += frame_box.astype(np.float)
                self.frames[frame][1] += 1
                self.frames[frame][2] += score

        for frame in range(mid_frame + 1, end_frame + 1):
            frame_box = mid_box * (end_frame - frame) / (end_frame - mid_frame) + end_box * (frame - mid_frame) / (end_frame - mid_frame)
            if frame not in self.frames:
                self.frames[frame] = [frame_box, 1, score]
            else:
                self.frames[frame][0] += frame_box.astype(np.float)
                self.frames[frame][1] += 1
                self.frames[frame][2] += score

        # Add middle frame
        if mid_frame not in self.frames:
            self.frames[mid_frame] = [mid_box.astype(np.float), 1, score]
        else:
            self.frames[mid_frame][0] += mid_box.astype(np.float)
            self.frames[mid_frame][1] += 1
            self.frames[mid_frame][2] += score

    def add_node(self, node):
        # self.nodes.append(node)
        self.update_frames(node)
        # self._volatile_memory()

    def _volatile_memory(self):
        if len(self.nodes) > self._max_num_node:
            for i in range(int(self._max_num_node/2)):
                del self.nodes[i]


class Tracks:
    '''
    Track set. It contains all the tracks and manage the tracks. it has the following information
    1) tracks. the set of tracks
    2) keep the previous image and features
    '''
    def __init__(self):
        self.tracks = list() # the set of tracks
        self.max_drawing_track = 10

    def __getitem__(self, item):
        return self.tracks[item]

    def append(self, track):
        self.tracks.append(track)

    def get_track_by_id(self, id):
        for t in self.tracks:
            if t.id == id:
                return t
        return None

    def one_frame_pass(self):
        keep_track_set = list()
        for i, t in enumerate(self.tracks):
            t.add_age()
            if t.age < t._max_age:
                keep_track_set.append(i)

        self.tracks = [self.tracks[i] for i in keep_track_set]

    def show(self, image):
        h, w, _ = image.shape

        # draw rectangle
        for t in self.tracks:
            if len(t.nodes) > 0 and t.age<2:
                b = t.nodes[-1].box
                image = cv2.putText(image, str(t.id), (int(b[0]*w),int((b[1])*h)), cv2.FONT_HERSHEY_SIMPLEX, 1, t.color, 3)
                image = cv2.rectangle(image, (int(b[0]*w),int((b[1])*h)), (int((b[0]+b[2])*w), int((b[1]+b[3])*h)), t.color, 2)

        # draw line
        for t in self.tracks:
            if t.age > 1:
                continue
            if len(t.nodes) > self.max_drawing_track:
                start = len(t.nodes) - self.max_drawing_track
            else:
                start = 0
            for n1, n2 in zip(t.nodes[start:], t.nodes[start+1:]):
                c1 = (int((n1.box[0] + n1.box[2]/2.0)*w), int((n1.box[1] + n1.box[3])*h))
                c2 = (int((n2.box[0] + n2.box[2] / 2.0) * w), int((n2.box[1] + n2.box[3]) * h))
                image = cv2.line(image, c1, c2, t.color, 2)

        return image


def update_tracks(tracks, tube, arg):
    mid_frame = tube[0].astype(np.int)
    mid_box = tube[1:5]
    end_frame = tube[5].astype(np.int)
    end_box = tube[6:10]
    start_frame = tube[10].astype(np.int)
    start_box = tube[11:15]
    score = tube[15]

    def get_center(box):

        return np.array([(box[0] + box[2])/2, (box[1] + box[3])/2])

    tube_direction = get_center(end_box) - get_center(start_box)

    assert start_frame <= mid_frame and mid_frame <= end_frame

    # Pre-compute all inter frame_boxs in this tube
    back_frames = list(range(start_frame, mid_frame))
    front_frames = list(range(mid_frame + 1, end_frame + 1))
    all_tube_frames = back_frames + front_frames + [mid_frame]
    # ! CAUTION: all_tube_frames is not sorted, mid_frame is the last one

    back_start_coef = (mid_frame - back_frames) / (mid_frame - start_frame)
    back_mid_coef = (back_frames - start_frame) / (mid_frame - start_frame)
    front_mid_coef = (end_frame - front_frames) / (end_frame - mid_frame)
    front_end_coef = (front_frames - mid_frame) / (end_frame - mid_frame)
    frame_boxs = np.concatenate((np.outer(back_start_coef, start_box), np.outer(front_end_coef, end_box))) + \
            np.outer(np.concatenate((back_mid_coef, front_mid_coef)), mid_box)
    frame_boxs = np.concatenate((frame_boxs, mid_box[None]))

    tube_frame_num = len(frame_boxs)

    # Above code computes bboxes in tube of corresponding frames
    # Equal to:
    # back_frame_boxs = np.outer((mid_frame - back_frames) / (mid_frame - start_frame), start_box) + \
    #         np.outer((back_frames - start_frame) / (mid_frame - start_frame), mid_box)
    # front_frame_boxs = np.outer((end_frame - front_frames) / (end_frame - mid_frame), mid_box) + \
    #         np.outer((front_frames - mid_frame) / (end_frame - mid_frame), end_box)
    # frame_boxs = np.concatenate((back_frame_boxs, front_frame_boxs))

    # Preallocate array of bboxes in track
    track_boxs = np.zeros_like(frame_boxs)

    max_idx, max_iou = -1, -1
    for idx, track in enumerate(tracks):
        iou = [0, 0]

        has_frame = [(frame in track.frames) for frame in all_tube_frames]
        if sum(has_frame) == 0:  # tube and track does not overlap
            continue

        # get the same length of area in the track that near to the tube
        overlap_frames = np.array(all_tube_frames)[np.where(has_frame)[0]]
        overlap_area = [min(overlap_frames), max(overlap_frames)]
        while overlap_area[1] - overlap_area[0] + 1 < tube_frame_num:
            if overlap_area[0] - 1 in track.frames:
                overlap_area[0] = overlap_area[0] - 1
            elif overlap_area[1] + 1 in track.frames:
                overlap_area[1] = overlap_area[1] + 1
            else:
                break
        # calculate the cos value
        track_direction = get_center(track.frames[overlap_area[1]][0] / track.frames[overlap_area[1]][1]) - \
                          get_center(track.frames[overlap_area[0]][0] / track.frames[overlap_area[0]][1])

        if np.linalg.norm(tube_direction) < arg.noise_dis:
            tube_direction = np.array([0, 0])
        if np.linalg.norm(track_direction) < arg.noise_dis:
            track_direction = np.array([0, 0])
        if np.linalg.norm(track_direction) * np.linalg.norm(tube_direction) > 0:
            angle_cos = np.dot(track_direction, tube_direction) / (np.linalg.norm(track_direction) * np.linalg.norm(tube_direction))
        else:
            angle_cos = 1

        # calculate the IoU
        for i, frame in enumerate(all_tube_frames):
            if has_frame[i]:
                track_boxs[i] = track.frames[frame][0] / track.frames[frame][1]

        iou[0] = sum(bbox_iou(frame_boxs, track_boxs)[has_frame])
        iou[1] = sum(has_frame)

        if iou[0] / iou[1] > arg.linking_min_iou + 0.2:
            angle_cos = 1

        # whether linking
        if iou[1] > 0 and iou[0] / iou[1] > max_iou and angle_cos > arg.cos_value:
            max_idx = idx
            max_iou = iou[0] / iou[1]

    if max_iou > arg.linking_min_iou:
        tracks[max_idx].update_frames(Node(tube))
    else:
        new_tracks(tracks, [tube])


def new_tracks(tracks, tubes):
    for tube in tubes:
        track = Track()
        track.add_node(Node(tube))
        tracks.append(track)


def final_processing(tracks, save_path):
    res = []
    assert len(tracks) != 0, 'No Tracks: ' + str(save_path)
    for track in tracks:
        cur_res = np.zeros((len(track.frames), 10))
        for i, (frame, bbox) in enumerate(track.frames.items()):
            cur_res[i, 0] = frame + 1
            cur_res[i, 2:6] = bbox[0] / bbox[1]
            cur_res[i, 6] = bbox[1]  # num of average bbox
            cur_res[i, 7] = bbox[2] / bbox[1]  # average score
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
    # ? return res or track


def archive_tracks(tracks, arch_tracks, cur_frame, forward_frames):
    track_ = []
    for track in tracks:
        max_frame = max(track.frames.keys())
        if (max_frame + 2 * forward_frames < cur_frame):
            arch_tracks.append(track)
        else:
            track_.append(track)

    return track_


def matching(tubes, arg, save_path=None, verbose=False):
    """
    tubes: All tubes in a video to match. (n, 15 + 1) [mid_frame, mid_box, front_frame, front_box, back_frame, back_box, value]
    save_path: File path to save formatted result.
    """
    tracks = []
    if not isinstance(tubes, np.ndarray):
        tubes = tubes.cpu().data.numpy()
    tubes = pd.DataFrame(tubes)
    tubes = tubes.astype({0: int, 5: int, 10: int})
    tubes_group = tubes.groupby(0)  # group by back_frame, i.e. start_frame

    arch_tracks = []
    for frame in sorted(tubes_group.indices.keys()):
        tubes_one_frame = tubes_group.get_group(frame).values

        for tube in tubes_one_frame:
            update_tracks(tracks, tube, arg)

        if verbose:
            print('{}\tFrame: {}\tTubes: {}\tCur tracks:{}\tArch tracks:{}'.format(\
                    datetime.now().time(), frame, len(tubes_one_frame), len(tracks), len(arch_tracks)))

        # Archive tracks 2*forward_frames frames away, they won't be useful anymore
        tracks = archive_tracks(tracks, arch_tracks, frame, arg.forward_frames * arg.frame_stride)

    arch_tracks.extend(tracks)
    tracks = arch_tracks
    final_processing(tracks, save_path)
    return tracks
