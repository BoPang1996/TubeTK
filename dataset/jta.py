import torch.utils.data as data
from PIL import Image, ImageFile

from dataset.augmentation import SSJAugmentation
from dataset.Parsers.JTA import GTParser_JTA
ImageFile.LOAD_TRUNCATED_IMAGES = True


class JTATrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''
    def __init__(self,
                 jta_root,
                 epoch,
                 arg,
                 transform=SSJAugmentation,
                 ):
        # 1. init all the variables
        self.jta_root = jta_root
        self.transform = transform(size=arg.img_size, type='train')
        self.epoch = epoch

        # 2. init GTParser
        self.parser = GTParser_JTA(self.jta_root, 'train', forward_frames=arg.forward_frames,
                                   frame_stride=arg.frame_stride, min_vis=arg.min_visibility,
                                   value_range=arg.value_range)

    def __getitem__(self, item):
        item = item % len(self.parser)
        image, img_meta, tubes, labels, start_frame = self.parser[item]

        while image is None:
            item += 100
            image, img_meta, tubes, labels, start_frame = self.parser[item % len(self.parser)]

            print('None processing.')

        if self.transform is None:
            return image, img_meta, tubes, labels, start_frame
        else:
            image, img_meta, tubes, labels, start_frame = self.transform(image, img_meta, tubes, labels, start_frame)
            return image, img_meta, tubes, labels, start_frame

    def __len__(self):
        return len(self.parser) * self.epoch

