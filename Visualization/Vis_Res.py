import cv2
import os
import argparse
import configparser
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


def get_seq_info_from_file_mot(seqName, dataDir):
    seqFolder = os.path.join(dataDir, seqName)
    seqInfoFile = os.path.join(dataDir, seqName, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seqInfoFile)

    imgFolder = config.get('Sequence', 'imDir')
    frameRate = config.getfloat('Sequence', 'frameRate')
    F = config.getint('Sequence', 'seqLength')
    imWidth = config.getint('Sequence', 'imWidth')
    imHeight = config.getint('Sequence', 'imHeight')
    imgExt = config.get('Sequence', 'imExt')

    return imgFolder, frameRate, imWidth, imHeight


def vis_one_video(res_file, frame_rate, img_width, img_height, img_dir, output_name):

    try:
        res = np.loadtxt(res_file, delimiter=',')
    except:
        res = np.loadtxt(res_file, delimiter=' ')
    res[:, 4:6] += res[:, 2:4]
    res = pd.DataFrame(res)
    res = res.replace([np.inf, -np.inf], np.nan)
    res = res.dropna()

    res_group = res.groupby(0)

    vid_writer = cv2.VideoWriter(output_name,
                                 cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (img_width, img_height))

    img_names = natsorted(os.listdir(img_dir))

    color_dict = {}
    for i, img_name in tqdm(enumerate(img_names), ncols=20):
        img = cv2.imread(os.path.join(img_dir, img_name))
        frame = int(os.path.splitext(img_name)[0])
        if frame not in res_group.groups.keys():
            vid_writer.write(img)
            continue
        bboxes = res_group.get_group(frame).values
        for bbox in bboxes:
            if bbox[1] in color_dict:
                color = color_dict[bbox[1]]
            else:
                color = np.round(np.random.rand(3) * 255)
                color_dict[bbox[1]] = color
            cv2.rectangle(img, tuple(bbox[4:6].astype(int)), tuple(bbox[2:4].astype(int)), color=color, thickness=3)
            cv2.putText(img, str(bbox[6]) + ' ' + str(bbox[7])[0:5],
                        tuple(bbox[2:4].astype(int)), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
        vid_writer.write(img)
    vid_writer.release()


def vis_res(args):
    try:
        os.makedirs(args.output_dir)
    except:
        pass

    if args.video_list is not None:
        video_list = open(args.video_list).readlines()
        video_list = [x.strip() for x in video_list]
    else:
        video_list = os.listdir(args.res_dir)
        video_list = [x for x in video_list if x.endswith('txt')]
        video_list = [os.path.splitext(x)[0] for x in video_list]
        video_list = [x for x in video_list if os.path.exists(os.path.join(args.data_dir, args.mode, x))]

    for vid in video_list:
        print('Processing {}'.format(vid))
        res_file = os.path.join(args.res_dir, vid + '.txt')
        if not os.path.exists(res_file):
            res_file = os.path.join(args.res_dir, vid, 'gt.txt')

        if 'JTA' not in args.output_dir:
            img_folder, frame_rate, img_width, img_height = get_seq_info_from_file_mot(vid, os.path.join(args.data_dir,
                                                                                                         args.mode))
        else:
            img_folder = ''
            frame_rate = 30
            img_width = 1920
            img_height = 1080

        img_dir = os.path.join(args.data_dir, args.mode, vid, img_folder)
        vis_one_video(res_file, frame_rate, img_width, img_height, img_dir, os.path.join(args.output_dir, vid + '.avi'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_dir', default='./link_res', type=str, help="path of the predicted tracks (saved as .txt)")
    parser.add_argument('--output_dir', default='./vis_video', type=str, help='where to save the output video')
    parser.add_argument('--data_dir', default='../data', type=str, help="input dataset path")
    parser.add_argument('--mode', default='test', type=str, help='vis the train or test set')
    parser.add_argument('--video_list', default='./seqmaps/MOT17_test.txt', type=str,
                        help='List for videos to visualize, None for all in res_dir')
    args = parser.parse_args()
    vis_res(args)
