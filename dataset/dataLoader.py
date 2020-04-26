from dataset.mot17 import MOT17TrainDataset, MOT17TestDataset
from dataset.jta import JTATrainDataset
from dataset.mot17jta import MOT17JTATrainDataset
import torch
try:
    import moxing.pytorch as mox
except:
    pass
import os


class Data_Loader_MOT():
    def __init__(self,
                 batch_size,
                 num_workers,
                 input_path,
                 model_arg,
                 train_epoch,
                 test_epoch,
                 dataset,
                 test_type='test',
                 test_seq=None):

        self.BATCH_SIZE = batch_size
        self.num_workers = num_workers

        def my_collate(batch):
            imgs = torch.stack([torch.tensor(item[0]) for item in batch], 0)
            img_meta = [item[1] for item in batch]
            tubes = [item[2] for item in batch]
            labels = [item[3] for item in batch]
            start_frame = [item[4] for item in batch]
            return imgs, img_meta, tubes, labels, start_frame

        if dataset == 'MOT17':
            print('MOT17 data')
            self.training_set = MOT17TrainDataset(mot_root=input_path, epoch=train_epoch, arg=model_arg)
            self.validation_set = MOT17TestDataset(mot_root=input_path, type=test_type, test_seq=test_seq,
                                                   epoch=test_epoch, arg=model_arg)
        elif dataset == 'JTA':
            print('JTA data')
            self.training_set = JTATrainDataset(jta_root=input_path, epoch=train_epoch, arg=model_arg)
            self.validation_set = None
        elif dataset == 'MOT17JTA':
            print('MOT17JTA data')
            self.training_set = MOT17JTATrainDataset(mot17_root=input_path[0], mot15_root=input_path[1],
                                                     jta_root=input_path[2], epoch=train_epoch, arg=model_arg)
            self.validation_set = None
        else:
            raise NotImplementedError

        # train loader
        if int(os.environ["RANK"]) == 0:
            print('==> Training data :', len(self.training_set))
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_set)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.training_set,
            batch_size=self.BATCH_SIZE,
            collate_fn=my_collate,
            num_workers=self.num_workers,
            pin_memory=True, sampler=train_sampler)

        # val loader
        if self.validation_set is not None:
            if int(os.environ["RANK"]) == 0:
                print('==> Validation data :', len(self.validation_set))
            val_sampler = torch.utils.data.distributed.DistributedSampler(self.validation_set)
            self.test_loader = torch.utils.data.DataLoader(
                dataset=self.validation_set,
                batch_size=self.BATCH_SIZE,
                collate_fn=my_collate,
                num_workers=self.num_workers,
                pin_memory=True, sampler=val_sampler)

