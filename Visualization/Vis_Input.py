import os
import dataset.augmentation as argument
import random
import cv2
import torch


class VisArgumentation(object):
    def __init__(self, size=896, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = argument.Compose([
            argument.AddMeans(self.mean),
            argument.ToCV2(),
            argument.ToAbsCoords()
        ])

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        return self.augment(images, img_meta, tubes, labels, start_frame)


def get_color():
    colors = [[i for i in range(0, 250, 20)],
              [i for i in range(0, 250, 20)],
              [i for i in range(0, 250, 20)]]
    for i in range(3):
        random.shuffle(colors[i])
    colors = list(zip(*colors))
    return colors


def get_inter_box(start_box, end_box, inter_id, end_id):
    if end_id == 0:
        return start_box
    return start_box * (end_id - inter_id) / end_id + end_box * inter_id / end_id


def tubes2bbox(tubes, colors, tube_len, stride):
    bboxes = [[] for _ in range(tube_len)]
    for i, tube in enumerate(tubes):
        color = colors[i % len(colors)]
        mid_frame = tube[4] / stride
        back_frame = (tube[4] - tube[10]) / stride
        front_frame = (tube[4] + tube[5]) / stride

        mid_bbox = tube[0:4]
        back_bbox = tube[11: 15] + mid_bbox
        front_bbox = tube[6: 10] + mid_bbox

        for f in range(int(back_frame), int(mid_frame)):
            bboxes[f].append([get_inter_box(back_bbox, mid_bbox, f - back_frame, mid_frame - back_frame), color])
        for f in range(int(mid_frame), int(front_frame + 1)):
            bboxes[f].append([get_inter_box(mid_bbox, front_bbox, f - mid_frame, front_frame - mid_frame), color])
        # bboxes[int(front_frame)].append([front_bbox, color])
    return bboxes


def vis_input(imgs, img_metas, gt_bboxes, gt_labels, start_frame, stride, out_folder):
    imgs_v, img_metas_v, gt_bboxes_v, gt_labels_v, start_frame_v = \
        VisArgumentation()(imgs[0], img_metas[0], gt_bboxes[0], gt_labels[0], start_frame[0])

    for f in range(len(imgs_v)):
        imgs_c = imgs_v.copy()
        f_tubes = []
        for tube in gt_bboxes_v:
            if round(tube[4]) == f * stride:
                f_tubes.append(tube)
        bboxes = tubes2bbox(f_tubes, get_color(), len(imgs_v), stride=stride)
        for i in range(len(imgs_c)):
            f_bboxes = bboxes[i]
            for bbox in f_bboxes:
                cv2.rectangle(imgs_c[i], tuple(bbox[0][0:2]), tuple(bbox[0][2:4]), bbox[1], 1)
            cv2.imwrite(os.path.join(out_folder, str(i) + '.jpg'), imgs_c[i])


def tubes2bbox_out(tubes, colors, tube_len, stride):

    bboxes = [[] for _ in range(tube_len)]
    for i, tube in enumerate(tubes):
        color = colors[i % len(colors)]
        mid_frame = tube[0] / stride
        back_frame = tube[10] / stride
        front_frame = tube[5] / stride

        mid_bbox = tube[1:5]
        back_bbox = tube[11: 15]
        front_bbox = tube[6: 10]

        for f in range(int(back_frame), int(mid_frame)):
            bboxes[f].append([get_inter_box(back_bbox, mid_bbox, f - back_frame, mid_frame - back_frame), color])
        for f in range(int(mid_frame), int(front_frame + 1)):
            bboxes[f].append([get_inter_box(mid_bbox, front_bbox, f - mid_frame, front_frame - mid_frame), color])
    return bboxes


def vis_output(imgs, img_metas, gt_bboxes, stride, out_folder):
    no_use = torch.tensor(gt_bboxes)
    imgs, img_metas, _, gt_labels, start_frame = \
        VisArgumentation()(imgs, img_metas, no_use, torch.tensor(1), torch.tensor(1))

    gt_bboxes[:, [1, 3, 6, 8, 11, 13]] /= img_metas['img_shape'][2] / img_metas['pad_percent'][0] / img_metas['value_range'] / 1024
    gt_bboxes[:, [2, 4, 7, 9, 12, 14]] /= img_metas['img_shape'][1] / img_metas['pad_percent'][1] / img_metas['value_range'] / 768
    gt_bboxes = gt_bboxes.data.numpy()
    for f in range(len(imgs)):
        imgs_c = imgs.copy()
        f_tubes = []
        for tube in gt_bboxes:
            if round(tube[0]) == f * stride:
                f_tubes.append(tube)
        bboxes = tubes2bbox_out(f_tubes, get_color(), len(imgs), stride=stride)
        write_folder = os.path.join(out_folder, img_metas['video_name'], str(img_metas['start_frame']), str(f))
        os.makedirs(write_folder, exist_ok=True)
        for i in range(len(imgs_c)):
            f_bboxes = bboxes[i]
            for bbox in f_bboxes:
                cv2.rectangle(imgs_c[i], tuple(bbox[0][0:2]), tuple(bbox[0][2:4]), bbox[1], 1)
            cv2.imwrite(os.path.join(write_folder, str(i) + '.jpg'), imgs_c[i])
