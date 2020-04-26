import torch
from torchvision import transforms
import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        for i, t in enumerate(self.transforms):
            image, img_meta, tubes, labels, start_frame = \
                t(image, img_meta, tubes, labels, start_frame)
        return image, img_meta, tubes, labels, start_frame


class ConvertFromInts(object):
    def __call__(self, image, img_meta, tubes, labels, start_frame):
        return image.astype(np.float32), img_meta, tubes.astype(np.float32), labels, start_frame


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        image = image.astype(np.float32)
        image -= self.mean
        image /= 255
        return image, img_meta, tubes, labels, start_frame


class AddMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        image *= 255
        try:
            image += self.mean
        except:
            image += torch.tensor(self.mean).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, image.shape[1], image.shape[2], image.shape[3])
        return image, img_meta, tubes, labels, start_frame


class ToPercentCoords(object):
    def __call__(self, image, img_meta, tubes, labels, start_frame):
        frame, height, width, channels = image.shape
        tubes[:, [0, 2, 6, 8, 11, 13]] /= width
        tubes[:, [1, 3, 7, 9, 12, 14]] /= height
        tubes[:, [4, 5, 10]] /= img_meta['img_shape'][0]
        tubes *= img_meta['value_range']
        return image, img_meta, tubes, labels, start_frame


class ToAbsCoords(object):
    def __call__(self, image, img_meta, tubes, labels, start_frame):
        frame, height, width, channels = image.shape
        tubes[:, [0, 2, 6, 8, 11, 13]] *= width
        tubes[:, [1, 3, 7, 9, 12, 14]] *= height
        tubes[:, [4, 5, 10]] *= img_meta['img_shape'][0]
        tubes /= img_meta['value_range']
        return image, img_meta, tubes, labels, start_frame


class Resize(object):
    def __init__(self, size):
        self.size = size

    def imrescale(self, img, scale, return_scale=False, interpolation='bilinear'):
        """Resize image while keeping the aspect ratio.
        Args:
            img (ndarray): The input image.
            scale (float or tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by this
                factor, else if it is a tuple of 2 integers, then the image will
                be rescaled as large as possible within the scale.
            return_scale (bool): Whether to return the scaling factor besides the
                rescaled image.
            interpolation (str): Same as :func:`resize`.
        Returns:
            ndarray: The rescaled image.
        """

        def _scale_size(size, scale):
            """Rescale a size by a ratio.
            Args:
                size (tuple): w, h.
                scale (float): Scaling factor.
            Returns:
                tuple[int]: scaled size.
            """
            w, h = size
            return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)

        def imresize(img, size, return_scale=False, interpolation='bilinear'):
            """Resize image to a given size.
            Args:
                img (ndarray): The input image.
                size (tuple): Target (w, h).
                return_scale (bool): Whether to return `w_scale` and `h_scale`.
                interpolation (str): Interpolation method, accepted values are
                    "nearest", "bilinear", "bicubic", "area", "lanczos".
            Returns:
                tuple or ndarray: (`resized_img`, `w_scale`, `h_scale`) or
                    `resized_img`.
            """
            interp_codes = {
                'nearest': cv2.INTER_NEAREST,
                'bilinear': cv2.INTER_LINEAR,
                'bicubic': cv2.INTER_CUBIC,
                'area': cv2.INTER_AREA,
                'lanczos': cv2.INTER_LANCZOS4
            }

            h, w = img.shape[:2]
            resized_img = cv2.resize(
                img, size, interpolation=interp_codes[interpolation])
            if not return_scale:
                return resized_img
            else:
                w_scale = size[0] / w
                h_scale = size[1] / h
                return resized_img, w_scale, h_scale

        h, w = img.shape[:2]
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(
                    'Invalid scale {}, must be positive.'.format(scale))
            scale_factor = scale
        elif isinstance(scale, list):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w),
                               max_short_edge / min(h, w))
        else:
            raise TypeError(
                'Scale must be a number or tuple of int, but got {}'.format(
                    type(scale)))
        new_size = _scale_size((w, h), scale_factor)
        rescaled_img = imresize(img, new_size, interpolation=interpolation)
        if return_scale:
            return rescaled_img, scale_factor
        else:
            return rescaled_img

    def impad(self, img, shape, pad_val=0):
        """Pad an image to a certain shape.
        Args:
            img (ndarray): Image to be padded.
            shape (tuple): Expected padding shape.
            pad_val (number or sequence): Values to be filled in padding areas.
        Returns:
            ndarray: The padded image.
        """
        if not isinstance(pad_val, (int, float)):
            assert len(pad_val) == img.shape[-1]
        if len(shape) < len(img.shape):
            shape = shape + [img.shape[-1]]
        assert len(shape) == len(img.shape)
        for i in range(len(shape) - 1):
            assert shape[i] >= img.shape[i]
        pad = np.empty(shape, dtype=img.dtype)
        pad[...] = pad_val
        pad[:img.shape[0], :img.shape[1], ...] = img
        return pad

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        new_images = []
        for i in range(len(images)):
            # new_images.append(cv2.resize(images[i], (self.size, self.size)))
            image = self.imrescale(images[i], self.size)
            size_before_pad = image.shape[:2]
            image = self.impad(image, self.size)
            size_after_pad = image.shape[:2]
            new_images.append(image)
        tubes[:, [0, 2, 6, 8, 11, 13]] *= size_before_pad[1]/size_after_pad[1]
        tubes[:, [1, 3, 7, 9, 12, 14]] *= size_before_pad[0]/size_after_pad[0]
        new_images = np.array(new_images)
        img_meta['pad_percent'] = [size_before_pad[1]/size_after_pad[1], size_before_pad[0]/size_after_pad[0]]

        return new_images, img_meta, tubes, labels, start_frame


class RandomSaturation(object):
    def __init__(self, lower=0.7, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image[:, :, :, 1] = image[:, :, :, 1] * alpha

        return image, img_meta, tubes, labels, start_frame


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image[:, :, :, 0] = image[:, :, :, 0] + delta
            image[:, :, :, 0][image[:, :, :, 0] > 360.0] -= 360.0
            image[:, :, :, 0][image[:, :, :, 0] < 0.0] += 360.0

        return image, img_meta, tubes, labels, start_frame


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            for i in range(len(image)):
                image[i] = shuffle(image[i])
        return image, img_meta, tubes, labels, start_frame


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if self.current == 'BGR' and self.transform == 'HSV':
            for i in range(len(image)):
                image[i] = cv2.cvtColor(image[i], cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            for i in range(len(image)):
                image[i] = cv2.cvtColor(image[i], cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, img_meta, tubes, labels, start_frame


class RandomContrast(object):
    def __init__(self, lower=0.7, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image = image*alpha
        return image, img_meta, tubes, labels, start_frame


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image = image + delta
        return image, img_meta, tubes, labels, start_frame


class RandomSampleCrop(object):
    """Crop
    Arguments:
        mode (float tuple): the min and max jaccard overlaps
    """

    def crop(self, images, tubes, labels, w, h, left, top):
        rect = np.array([int(left), int(top), int(left + w), int(top + h)])

        # cut the crop from the image
        images = images[:, rect[1]:rect[3], rect[0]:rect[2], :]

        # keep overlap with gt box IF center in sampled patch
        centers = (tubes[:, :2] + tubes[:, 2:4]) / 2.0

        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

        # mask in that both m1 and m2 are true
        mask = m1 * m2

        # have any valid boxes? try again if not
        if not mask.any():
            return None

        # take only matching gt boxes
        tubes = tubes[mask, :].copy()
        labels = labels[mask, :].copy()

        tubes[:, 6:10] = tubes[:, 6:10] + tubes[:, 0:4]
        tubes[:, 11:15] = tubes[:, 11:15] + tubes[:, 0:4]

        # should we use the box left and top corner or the crop's
        tubes[:, [0,1,6,7,11,12]] = np.maximum(tubes[:, [0,1,6,7,11,12]], rect[[0,1,0,1,0,1]])
        # adjust to crop (by substracting crop's left,top)
        tubes[:, [0,1,6,7,11,12]] -= rect[[0,1,0,1,0,1]]

        tubes[:, [2,3,8,9,13,14]] = np.minimum(tubes[:, [2,3,8,9,13,14]], rect[[2,3,2,3,2,3]])
        # adjust to crop (by substracting crop's left,top)
        tubes[:, [2,3,8,9,13,14]] -= rect[[0,1,0,1,0,1]]

        tubes[:, 6:10] = tubes[:, 6:10] - tubes[:, 0:4]
        tubes[:, 11:15] = tubes[:, 11:15] - tubes[:, 0:4]
        return images, tubes, labels

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        height, width, _ = images[0].shape

        while True:
            w = random.uniform(0.85 * width, width)
            h = random.uniform(0.85 * height, height)

            # aspect ratio constraint b/t .5 & 2
            # if h / w < 0.5 or h / w > 2:
            #     continue

            left = random.uniform(width - w)
            top = random.uniform(height - h)

            img_meta['img_shape'] = [img_meta['img_shape'][0], h, w]
            res_pre = self.crop(images, tubes, labels, w, h, left, top)
            if res_pre is None:
                continue
            images, tubes, labels = res_pre
            return images, img_meta, tubes, labels, start_frame


class Expand(object):
    def __init__(self, mean=(104, 117, 123)):
        self.mean = mean

    def expand(self, image, frames, height, width, depth, ratio, left, top):
        expand_image = np.zeros(
            (int(frames), int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :, :] = self.mean
        expand_image[:, int(top):int(top + height), int(left):int(left + width)] = image
        return expand_image

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            return images, img_meta, tubes, labels, start_frame

        frames, height, width, depth = images.shape
        ratio = random.uniform(1, 1.2)  # new: adjust the max_expand to 2.0 (orgin is 4.0)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        images = self.expand(images, frames, height, width, depth, ratio, left, top)
        img_meta['img_shape'] = [img_meta['img_shape'][0], images.shape[1], images.shape[2]]
        tubes = tubes.copy()
        tubes[:, [0, 1, 2, 3]] += (int(left), int(top), int(left), int(top))

        return images, img_meta, tubes, labels, start_frame


class RandomMirror(object):
    def mirror(self, images, tubes):
        _, _, width, _ = images.shape
        images = np.array(images[:, :, ::-1])
        tubes2 = tubes.copy()
        tubes2[:, 0] = width - tubes[:, 2]
        tubes2[:, 2] = width - tubes[:, 0]
        tubes2[:, [6, 8, 11, 13]] = -1 * tubes[:, [8, 6, 13, 11]]
        return images, tubes2

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        if random.randint(2):
            res = self.mirror(images, tubes)
            images = res[0]
            tubes = res[1]

        return images, img_meta, tubes, labels, start_frame


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        images = images.copy()
        images, img_meta, tubes, labels, start_frame = \
            self.rand_brightness(images, img_meta, tubes, labels, start_frame)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])

        images, img_meta, tubes, labels, start_frame = distort(images, img_meta, tubes, labels, start_frame)

        return self.rand_light_noise(images, img_meta, tubes, labels, start_frame)


class ToTensor(object):
    def __call__(self, images, img_meta, tubes, labels, start_frame):
        images = images
        images = torch.from_numpy(images.astype(np.float32)).permute(3, 0, 1, 2)

        tubes = torch.from_numpy(tubes.astype(np.float32))
        labels = torch.from_numpy(labels)

        start_frame = torch.tensor(start_frame)

        return images, img_meta, tubes, labels, start_frame


class ToCV2(object):
    def __call__(self, images, img_meta, tubes, labels, start_frame):
        images = images.permute(1, 2, 3, 0).int().cpu().data.numpy()
        tubes = tubes.float().cpu().data.numpy()
        labels = labels.cpu().data.numpy()
        start_frame = start_frame.cpu().data.numpy()

        return images, img_meta, tubes, labels, start_frame


class SSJAugmentation(object):
    def __init__(self, size=896, mean=(104, 117, 123), type='train'):
        self.mean = mean
        self.size = size
        if type == 'train':
            self.augment = Compose([
                ConvertFromInts(),
                PhotometricDistort(),
                Expand(self.mean),
                RandomSampleCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                ToTensor()
            ])
        elif type == 'test':
            self.augment = Compose([
                ConvertFromInts(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                ToTensor()
            ])
        else:
            raise NameError('config type is wrong, should be choose from (train, test)')

    def __call__(self, images, img_meta, tubes, labels, start_frame):
        return self.augment(images, img_meta, tubes, labels, start_frame)
