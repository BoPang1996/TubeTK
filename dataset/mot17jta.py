import torch.utils.data as data
from PIL import ImageFile
from dataset.Parsers.MOT17 import GTParser_MOT_17
from dataset.Parsers.JTA import GTParser_JTA
from dataset.augmentation import SSJAugmentation

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MOT17JTATrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''
    def __init__(self,
                 mot17_root,
                 mot15_root,
                 jta_root,
                 epoch,
                 arg,
                 transform=SSJAugmentation,
                 ):
        # 1. init all the variables
        self.mot17_root = mot17_root
        self.mot15_root = mot15_root
        self.jta_root = jta_root
        self.transform = transform(size=arg.img_size, type='train')
        self.epoch = epoch

        self.parsers = {}
        # 2. init GTParser
        self.parser_MOT17 = GTParser_MOT_17(self.mot17_root, 'train', forward_frames=arg.forward_frames,
                                            frame_stride=arg.frame_stride, min_vis=arg.min_visibility,
                                            value_range=arg.value_range)
        self.parsers['MOT17'] = self.parser_MOT17

        self.parser_JTA = GTParser_JTA(self.jta_root, 'train', forward_frames=arg.forward_frames,
                                       frame_stride=arg.frame_stride, min_vis=0.3,
                                       value_range=arg.value_range)
        self.parsers['JTA'] = self.parser_JTA

    def __getitem__(self, item):

        mot17 = True if item < len(self.parser_MOT17) * self.epoch else False
        if mot17:
            parser = self.parsers['MOT17']
            item = item % len(self.parser_MOT17)

        if not mot17:
            parser = self.parsers['JTA']
            item = (item - len(self.parser_MOT17) * self.epoch) % len(self.parser_JTA)

        image, img_meta, tubes, labels, start_frame = parser[item]
        while image is None:
            print('None processing.')
            item += 100
            image, img_meta, tubes, labels, start_frame = parser[item % len(parser)]

        if self.transform is None:
            return image, img_meta, tubes, labels, start_frame
        else:
            image, img_meta, tubes, labels, start_frame = self.transform(image, img_meta, tubes, labels, start_frame)
            return image, img_meta, tubes, labels, start_frame

    def __len__(self):
        return len(self.parser_MOT17) * self.epoch * 2


