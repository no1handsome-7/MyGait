import numpy as np
import random
import torchvision.transforms as T
import cv2
import math
from data import transform as base_transform
from utils import is_list, is_dict, get_valid_args


class NoOperation():
    def __call__(self, x):
        return x


class BaseSilTransform():
    def __init__(self, divsor=255.0, img_shape=None):
        self.divsor = divsor
        self.img_shape = img_shape

    def __call__(self, x):
        if self.img_shape is not None:
            s = x.shape[0]
            _ = [s] + [*self.img_shape]
            x = x.reshape(*_)
        return x / self.divsor


class BaseParsingCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        if x.max() == 255 or x.max() == 255.:
            return x / self.divsor
        else:
            return x / 1.0


class BaseSilCuttingTransform():
    def __init__(self, divsor=255.0, cutting=None):
        self.divsor = divsor
        self.cutting = cutting

    def __call__(self, x):
        if self.cutting is not None:
            cutting = self.cutting
        else:
            cutting = int(x.shape[-1] // 64) * 10
        if cutting != 0: 
            x = x[..., cutting:-cutting]
        return x / self.divsor


# class BaseRgbTransform():
#     def __init__(self, mean=None, std=None):
#         if mean is None:
#             mean = [0.485*255, 0.456*255, 0.406*255]
#         if std is None:
#             std = [0.229*255, 0.224*255, 0.225*255]
#         self.mean = np.array(mean).reshape((1, 3, 1, 1))
#         self.std = np.array(std).reshape((1, 3, 1, 1))

#     def __call__(self, x):
#         return (x - self.mean) / self.std
    
class BaseRgbTransform():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = [0.485*255, 0.456*255, 0.406*255]
        if std is None:
            std = [0.229*255, 0.224*255, 0.225*255]
        self.channel_num = len(mean)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def _to_channel_first(self, x):
        c = self.channel_num
        if x.ndim == 3:
            if x.shape[0] == c:       # CHW
                return x
            if x.shape[-1] == c:      # HWC -> CHW
                return x.transpose(2, 0, 1)
        elif x.ndim == 4:
            if x.shape[1] == c:       # TCHW
                return x
            if x.shape[-1] == c:      # THWC -> TCHW
                return x.transpose(0, 3, 1, 2)
        elif x.ndim == 5:
            if x.shape[2] == c:       # BTCHW
                return x
            if x.shape[-1] == c:      # BTHWC -> BTCHW
                return x.transpose(0, 1, 4, 2, 3)
        raise ValueError(
            f'Unsupported RGB input shape {x.shape}. Expected CHW/HWC, TCHW/THWC, or BTCHW/BTHWC with {c} channels.'
        )

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = self._to_channel_first(x)

        if x.ndim == 3:       # CHW
            mean = self.mean.reshape((self.channel_num, 1, 1))
            std = self.std.reshape((self.channel_num, 1, 1))
        elif x.ndim == 4:     # TCHW
            mean = self.mean.reshape((1, self.channel_num, 1, 1))
            std = self.std.reshape((1, self.channel_num, 1, 1))
        elif x.ndim == 5:     # BTCHW
            mean = self.mean.reshape((1, 1, self.channel_num, 1, 1))
            std = self.std.reshape((1, 1, self.channel_num, 1, 1))
        else:
            raise ValueError(f'Unsupported RGB input shape after layout conversion: {x.shape}')
        return np.ascontiguousarray((x - mean) / std)


# **************** Data Agumentation ****************


# class RandomHorizontalFlip(object):
#     def __init__(self, prob=0.5):
#         self.prob = prob

#     def __call__(self, seq):
#         if random.uniform(0, 1) >= self.prob:
#             return seq
#         else:
#             return seq[..., ::-1]


# class RandomErasing(object):
#     def __init__(self, prob=0.5, sl=0.05, sh=0.2, r1=0.3, per_frame=False):
#         self.prob = prob
#         self.sl = sl
#         self.sh = sh
#         self.r1 = r1
#         self.per_frame = per_frame

#     def __call__(self, seq):
#         if not self.per_frame:
#             if random.uniform(0, 1) >= self.prob:
#                 return seq
#             else:
#                 for _ in range(100):
#                     seq_size = seq.shape
#                     area = seq_size[1] * seq_size[2]

#                     target_area = random.uniform(self.sl, self.sh) * area
#                     aspect_ratio = random.uniform(self.r1, 1 / self.r1)

#                     h = int(round(math.sqrt(target_area * aspect_ratio)))
#                     w = int(round(math.sqrt(target_area / aspect_ratio)))

#                     if w < seq_size[2] and h < seq_size[1]:
#                         x1 = random.randint(0, seq_size[1] - h)
#                         y1 = random.randint(0, seq_size[2] - w)
#                         seq[:, x1:x1+h, y1:y1+w] = 0.
#                         return seq
#             return seq
#         else:
#             self.per_frame = False
#             frame_num = seq.shape[0]
#             ret = [self.__call__(seq[k][np.newaxis, ...])
#                    for k in range(frame_num)]
#             self.per_frame = True
#             return np.concatenate(ret, 0)


class RandomRotate(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            degree = random.uniform(-self.degree, self.degree)
            M1 = cv2.getRotationMatrix2D((dh // 2, dw // 2), degree, 1)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomPerspective(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            h, w = seq.shape[-2:]
            cutting = int(w // 44) * 10
            x_left = list(range(0, cutting))
            x_right = list(range(w - cutting, w))
            TL = (random.choice(x_left), 0)
            TR = (random.choice(x_right), 0)
            BL = (random.choice(x_left), h)
            BR = (random.choice(x_right), h)
            srcPoints = np.float32([TL, TR, BR, BL])
            canvasPoints = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            perspectiveMatrix = cv2.getPerspectiveTransform(
                np.array(srcPoints), np.array(canvasPoints))
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpPerspective(_[0, ...], perspectiveMatrix, (w, h))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq


class RandomAffine(object):
    def __init__(self, prob=0.5, degree=10):
        self.prob = prob
        self.degree = degree

    def __call__(self, seq):
        if random.uniform(0, 1) >= self.prob:
            return seq
        else:
            dh, dw = seq.shape[-2:]
            # rotation
            max_shift = int(dh // 64 * 10)
            shift_range = list(range(0, max_shift))
            pts1 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            pts2 = np.float32([[random.choice(shift_range), random.choice(shift_range)], [
                              dh-random.choice(shift_range), random.choice(shift_range)], [random.choice(shift_range), dw-random.choice(shift_range)]])
            M1 = cv2.getAffineTransform(pts1, pts2)
            # affine
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 2, 3, 1)
            seq = [cv2.warpAffine(_[0, ...], M1, (dw, dh))
                   for _ in np.split(seq, seq.shape[0], axis=0)]
            seq = np.concatenate([np.array(_)[np.newaxis, ...]
                                 for _ in seq], 0)
            if len(seq.shape) == 4:
                seq = seq.transpose(0, 3, 1, 2)
            return seq

# for gpf   
# -*- coding: utf-8 -*-
"""
GPF-aware data augmentation transforms for VOGait.

Drop this file into:
    OpenGait/opengait/data/transform.py  (append to existing, or import from a new file)

Or add to the existing transform registry.

These transforms handle the 12-channel GPF layout correctly:
  - Horizontal flip must negate normal_x (ch2) and flow_dx (ch8)
  - Random erasing respects the foreground mask (ch0)
"""

import random
import numpy as np


class RandomHorizontalFlip:
    def __init__(self, prob=0.5, flip_negate_indices=None):
        self.prob = float(prob)
        self.flip_negate_indices = list(flip_negate_indices) if flip_negate_indices else [2, 6]

    def __call__(self, seq):
        seq = np.asarray(seq)

        if random.random() > self.prob:
            return seq

        seq = seq[..., ::-1].copy()

        # GPF sequence: [T, C, H, W]
        if seq.ndim == 4:
            c = seq.shape[1]
            for idx in self.flip_negate_indices:
                if idx < c:
                    seq[:, idx] = -seq[:, idx]

        # Single GPF frame: [C, H, W]
        elif seq.ndim == 3 and seq.shape[0] <= 16:
            c = seq.shape[0]
            for idx in self.flip_negate_indices:
                if idx < c:
                    seq[idx] = -seq[idx]

        return seq


class RandomErasing:
    def __init__(self, prob=0.3, area_ratio_range=(0.02, 0.15), aspect_ratio_range=(0.3, 3.3)):
        self.prob = float(prob)
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range

    def __call__(self, seq):
        seq = np.asarray(seq)

        if random.random() > self.prob:
            return seq

        seq = seq.copy()

        # GPF sequence: [T, C, H, W]
        if seq.ndim == 4:
            t, c, h, w = seq.shape
            area = h * w
            target_area = random.uniform(*self.area_ratio_range) * area
            aspect = random.uniform(*self.aspect_ratio_range)

            rh = int(round((target_area * aspect) ** 0.5))
            rw = int(round((target_area / aspect) ** 0.5))
            rh = max(1, min(rh, h))
            rw = max(1, min(rw, w))

            top = random.randint(0, h - rh)
            left = random.randint(0, w - rw)

            frame_ratio = random.uniform(0.5, 1.0)
            num_frames = max(1, int(t * frame_ratio))
            frame_indices = random.sample(range(t), num_frames)

            seq[frame_indices, :, top:top + rh, left:left + rw] = 0.0

        # Silhouette sequence: [T, H, W]
        elif seq.ndim == 3 and seq.shape[0] > 16:
            t, h, w = seq.shape
            area = h * w
            target_area = random.uniform(*self.area_ratio_range) * area
            aspect = random.uniform(*self.aspect_ratio_range)

            rh = int(round((target_area * aspect) ** 0.5))
            rw = int(round((target_area / aspect) ** 0.5))
            rh = max(1, min(rh, h))
            rw = max(1, min(rw, w))

            top = random.randint(0, h - rh)
            left = random.randint(0, w - rw)

            frame_ratio = random.uniform(0.5, 1.0)
            num_frames = max(1, int(t * frame_ratio))
            frame_indices = random.sample(range(t), num_frames)

            seq[frame_indices, top:top + rh, left:left + rw] = 0.0

        return seq


class RandomTemporalCrop:
    """
    Randomly crop a contiguous temporal segment, adding temporal variation.
    Only applied during training when frames exceed min_frames.

    Input: seqs_data list, where seqs_data[0] has shape [T, C, H, W].
    """
    def __init__(self, min_frames=20, crop_ratio_range=(0.7, 1.0)):
        self.min_frames = int(min_frames)
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, seqs_data):
        seq = seqs_data[0]
        if not isinstance(seq, np.ndarray) or seq.ndim < 3:
            return seqs_data
        t = seq.shape[0]
        if t <= self.min_frames:
            return seqs_data
        crop_ratio = random.uniform(*self.crop_ratio_range)
        crop_len = max(self.min_frames, int(t * crop_ratio))
        if crop_len >= t:
            return seqs_data
        start = random.randint(0, t - crop_len)
        out = []
        for i, s in enumerate(seqs_data):
            if i == 0 and isinstance(s, np.ndarray):
                out.append(s[start:start + crop_len].copy())
            else:
                out.append(s)
        return out
        
# ******************************************

def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"

# **************** For LidarGait++ ****************
# Shen, et al: LidarGait++: Learning Local Features and Size Awareness from LiDAR Point Clouds for 3D Gait Recognition, CVPR2025

def normalize_point_cloud(batch_data):
    """Normalize the batch data using coordinates of the block centered at origin.
    
    Input:
        batch_data: BxNxC array
    Output:
        BxNxC array
    """
    centroids = np.mean(batch_data, axis=1, keepdims=True)  # shape: (B, 1, C)
    centered = batch_data - centroids
    scales = np.max(np.linalg.norm(centered, axis=2), axis=1, keepdims=True)  # shape: (B, 1)
    scales = scales.reshape(batch_data.shape[0], 1, 1)  # (B, 1, 1) for broadcasting
    return centered / scales


def dropout_point_cloud(batch_data, max_dropout_ratio=0.875, prob=0.2):
    """Randomly drop points in each point cloud.
    
    Input:
        batch_data: BxNx3 array
    Output:
        BxNx3 array, with dropped points replaced by the first point in each cloud.
    """
    if np.random.rand() >= prob:
        return batch_data
    B, N, C = batch_data.shape
    # 为每个点云生成一个 dropout_ratio（范围 0 ~ max_dropout_ratio）
    dropout_ratio = np.random.rand(B, 1) * max_dropout_ratio  # shape: (B, 1)
    random_matrix = np.random.rand(B, N)
    drop_mask = random_matrix <= dropout_ratio  # shape: (B, N)
    # 构造每个点云第一个点重复 N 次的数组，用于替换被 dropout 的点
    first_points = np.repeat(batch_data[:, :1, :], N, axis=1)
    return np.where(drop_mask[..., None], first_points, batch_data)
    
def shift_point_cloud(batch_data, shift_range=0.1, prob=0.2):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    if np.random.rand() >= prob:
        return batch_data
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, N,3))
    batch_data += shifts
    return batch_data


def scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25, prob=0.2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    if np.random.rand() >= prob:
        return batch_data
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def jitter_point_cloud(batch_data, std=0.01, clip=0.05, prob=0.2):
    if np.random.rand() >= prob:
        return batch_data
    B, N, C = batch_data.shape
    jittered_data = np.random.normal(loc=0.0, scale=std, size=(B, N, C))
    jittered_data = np.clip(jittered_data, -clip, clip)
    batch_data += jittered_data
    return batch_data

def flip_point_cloud_y(batch_data, prob=0.25):
    if np.random.rand() >= prob:
        return batch_data
    batch_data[:, :, 1] = -batch_data[:, :, 1]
    return batch_data


def getxyz(batch_data,col = 2,to_ground=False):
    B,N,C = batch_data.shape
    last_col = batch_data[:, :, col]
    result = last_col.reshape((B, N, 1))
    if to_ground:
        result -= result.min(axis=1,keepdims=True)
    return result

class PointCloudsTransform():
    def __init__(self, xyz_only=True, scale_aware=False, drop_prob=0, shift_prob=0, jit_prob=0,scale_prob=0, flip_prob=0):
        self.scale_aware = scale_aware
        self.xyz_only = xyz_only

        self.flip_prob, self.shift_prob, self.jit_prob, self.scale_prob, self.drop_prob = flip_prob, shift_prob, jit_prob, scale_prob, drop_prob 
    def __call__(self, points):
        if self.xyz_only:
            points = points[:,:,:3]
        
        heights = getxyz(points, col = 2, to_ground=True)
        points = normalize_point_cloud(points)

        points = flip_point_cloud_y(points, prob=self.flip_prob)
        points = shift_point_cloud(points, prob=self.shift_prob)
        points = jitter_point_cloud(points, prob=self.jit_prob)
        points = scale_point_cloud(points, prob=self.scale_prob)
        points = dropout_point_cloud(points, prob=self.drop_prob)

        if self.scale_aware:
            points = np.concatenate([points,heights],axis=-1)
        return points


# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023

class RandomPartDilate():
    def __init__(self, prob=0.5, top_range=(12, 16), bot_range=(36, 40)):
        self.prob = prob
        self.top_range = top_range
        self.bot_range = bot_range
        self.modes_and_kernels = {
            'RECT': [[5, 3], [5, 5], [3, 5]],
            'CROSS': [[3, 3], [3, 5], [5, 3]],
            'ELLIPSE': [[3, 3], [3, 5], [5, 3]]}
        self.modes = list(self.modes_and_kernels.keys())

    def __call__(self, seq):
        '''
            Using the image dialte and affine transformation to simulate the clorhing change cases.
        Input:
            seq: a sequence of silhouette frames, [s, h, w]
        Output:
            seq: a sequence of agumented frames, [s, h, w]
        '''
        if random.uniform(0, 1) >= self.prob:
                return seq
        else:
            mode = random.choice(self.modes)
            kernel_size = random.choice(self.modes_and_kernels[mode])
            top = random.randint(self.top_range[0], self.top_range[1])
            bot = random.randint(self.bot_range[0], self.bot_range[1])

            seq = seq.transpose(1, 2, 0) # [s, h, w] -> [h, w, s]
            _seq_ = seq.copy()
            _seq_ = _seq_[top:bot, ...]
            _seq_ = self.dilate(_seq_, kernel_size=kernel_size, mode=mode)
            seq[top:bot, ...] = _seq_
            seq = seq.transpose(2, 0, 1) # [h, w, s] -> [s, h, w]
            return seq

    def dilate(self, img, kernel_size=[3, 3], mode='RECT'):
        '''
            MORPH_RECT, MORPH_CROSS, ELLIPSE
        Input:
            img: [h, w]
        Output:
            img: [h, w]
        '''
        assert mode in ['RECT', 'CROSS', 'ELLIPSE']
        kernel = cv2.getStructuringElement(getattr(cv2, 'MORPH_'+mode), kernel_size)
        dst = cv2.dilate(img, kernel)
        return dst

class RandomPartBlur():
    def __init__(self, prob=0.5, top_range=(9, 20), bot_range=(29, 40), per_frame=False):
        self.prob = prob
        self.top_range = top_range
        self.bot_range = bot_range
        self.per_frame = per_frame

    def __call__(self, seq):
        '''
        Input:
            seq: a sequence of silhouette frames, [s, h, w]
        Output:
            seq: a sequence of agumented frames, [s, h, w]
        '''
        if not self.per_frame:
            if random.uniform(0, 1) >= self.prob:
                return seq
            else:
                top = random.randint(self.top_range[0], self.top_range[1])
                bot = random.randint(self.bot_range[0], self.bot_range[1])

                seq = seq.transpose(1, 2, 0) # [s, h, w] -> [h, w, s]
                _seq_ = seq.copy()
                _seq_ = _seq_[top:bot, ...]
                _seq_ = cv2.GaussianBlur(_seq_, ksize=(3, 3), sigmaX=0)
                _seq_ = (_seq_ > 0.2).astype(np.float)
                seq[top:bot, ...] = _seq_
                seq = seq.transpose(2, 0, 1) # [h, w, s] -> [s, h, w]

            return seq
        else:
            self.per_frame = False
            frame_num = seq.shape[0]
            ret = [self.__call__(seq[k][np.newaxis, ...]) for k in range(frame_num)]
            self.per_frame = True
            return np.concatenate(ret, 0)

def DA4GaitSSB(
    cutting = None,
    ra_prob = 0.2,
    rp_prob = 0.2,
    rhf_prob = 0.5,
    rpd_prob = 0.2,
    rpb_prob = 0.2,
    top_range = (9, 20),
    bot_range = (39, 50),
):
    transform = T.Compose([
            RandomAffine(prob=ra_prob),
            RandomPerspective(prob=rp_prob),
            BaseSilCuttingTransform(cutting=cutting),
            RandomHorizontalFlip(prob=rhf_prob),
            RandomPartDilate(prob=rpd_prob, top_range=top_range, bot_range=bot_range),
            RandomPartBlur(prob=rpb_prob, top_range=top_range, bot_range=bot_range),
    ])
    return transform

# **************** For pose-based methods ****************
class RandomSelectSequence(object):
    """
    Randomly select different subsequences
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            raise ValueError("The sequence length of data is too short, which does not meet the requirements.")
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    """
    Select center subsequence
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0]/2) - (self.sequence_length / 2))
        except ValueError:
            raise ValueError("The sequence length of data is too short, which does not meet the requirements.")
        end = start + self.sequence_length
        return data[start:end]


class MirrorPoses(object):
    """
    Performing Mirror Operations
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if np.random.random() <= self.prob:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class NormalizeEmpty(object):
    """
    Normliza Empty Joint
    """
    def __call__(self, data):
        frames, joints = np.where(data[:, :, 0] == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = np.mean(data[frame], axis=0)
            data[frame, joint, 0] = center_of_gravity[0]
            data[frame, joint, 1] = center_of_gravity[1]
            data[frame, joint, 2] = 0
        return data


class RandomMove(object):
    """
    Move: add Random Movement to each joint
    """
    def __init__(self,random_r =[4,1]):
        self.random_r = random_r
    def __call__(self, data):
        noise = np.zeros(3)
        noise[0] = np.random.uniform(-self.random_r[0], self.random_r[0])
        noise[1] = np.random.uniform(-self.random_r[1], self.random_r[1])
        data += np.tile(noise,(data.shape[0], data.shape[1], 1))
        return data


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class FlipSequence(object):
    """
    Temporal Fliping
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data,axis=0).copy()
        return data


class InversePosesPre(object):
    '''
    Left-right flip of skeletons
    '''
    def __init__(self, probability=0.5, joint_format='coco'):
        self.probability = probability
        if joint_format == 'coco':
            self.invers_arr = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        elif joint_format in ['alphapose', 'openpose']:
            self.invers_arr = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
        else:
            raise ValueError("Invalid joint_format.")
            

    def __call__(self, data):
        for i in range(len(data)):
            if np.random.random() <= self.probability:
                data[i]=data[i,self.invers_arr,:]
        return data


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.25):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, self.std, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class GaitTRMultiInput(object):
    def __init__(self, joint_format='coco',):
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((5, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, 0]
        # Velocity
        for i in range(T - 2):
            data_new[2, :, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[3, :, i, :] = data[:, i + 2, :] - data[:, i, :]
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[4, :, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        
        I, C, T, V = data_new.shape
        data_new = data_new.reshape(I*C, T, V)
        # (C T V) -> (T V C)
        data_new = np.transpose(data_new, (1, 2, 0))

        return data_new


class GaitGraphMultiInput(object):
    def __init__(self, center=0, joint_format='coco'):
        self.center = center
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
        else:
            raise ValueError("Invalid joint_format.")

    def __call__(self, data):
        T, V, C = data.shape
        x_new = np.zeros((T, V, 3, C + 2))
        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]
        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]
        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        # Angles
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(x_new[:, :, 2, i], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = np.arccos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]
        return x_new

class GaitGraph1Input(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (C T V)
        data = np.transpose(data, (2, 0, 1))
        return data[...,np.newaxis]

class SkeletonInput(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (T C V)
        data = np.transpose(data, (0, 2, 1))
        return data[...,np.newaxis]

class TwoView(object):
    def __init__(self,trf_cfg):
        assert is_list(trf_cfg)
        self.transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    def __call__(self, data):
        return np.concatenate([self.transform(data), self.transform(data)], axis=1)


class MSGGTransform():
    def __init__(self, joint_format="coco"):
        if joint_format == "coco": #17
            self.mask=[6,8,14,12,7,13,5,10,16,11,9,15]
        elif joint_format in ['alphapose', 'openpose']: #18
            self.mask=[2,3,9,8,6,12,5,4,10,11,7,13]
        else:
            raise ValueError("Invalid joint_format.")
        
    def __call__(self, x):
        result=x[...,self.mask,:].copy()
        return result

# **************** For GaitOracle ****************

class BaseRgbNormalizeTransform(BaseRgbTransform):
    """Alias for RGB normalization used by GaitOracle configs."""
    pass


class NightSimulationRgbTransform(BaseRgbNormalizeTransform):
    """RGB normalization with random night-condition simulation augmentation.

    Use in trainer_cfg ONLY (evaluator_cfg should keep BaseRgbNormalizeTransform).

    Simulates nighttime degradation on daytime RGB frames by:
      1) Gamma correction  — power-law darkening (low ambient light)
      2) Brightness scaling — global dimming
      3) Color temperature  — warm (streetlight) / cool (LED/moonlight) shift
      4) Contrast reduction — night scenes have compressed dynamic range
      5) Gaussian noise     — sensor noise in low-light conditions

    After augmentation, applies standard ImageNet-style normalization.

    Config example (lrgait.yaml trainer_cfg.transform):
        - type: NightSimulationRgbTransform
          prob: 0.3
          gamma_range: [1.8, 3.5]
          brightness_range: [0.05, 0.4]
          noise_std: 5.0
    """

    def __init__(self, prob=0.3, gamma_range=None, brightness_range=None,
                 noise_std=5.0, contrast_range=None,
                 color_temp_prob=0.3, color_temp_range=None,
                 mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.night_prob = float(prob)
        self.gamma_range = tuple(gamma_range) if gamma_range else (1.8, 3.5)
        self.brightness_range = tuple(brightness_range) if brightness_range else (0.05, 0.4)
        self.noise_std = float(noise_std)               # in [0, 255] scale
        self.contrast_range = tuple(contrast_range) if contrast_range else (0.3, 0.7)
        self.color_temp_prob = float(color_temp_prob)
        self.color_temp_range = tuple(color_temp_range) if color_temp_range else (0.8, 1.2)

    def _apply_night_sim(self, x):
        """Apply night simulation to channel-first data in [0, 255].

        Args:
            x: np.ndarray, CHW / TCHW / BTCHW, float32, [0, 255]
        Returns:
            np.ndarray, same shape, float32, [0, 255]
        """
        # --- work in [0, 1] ---
        x_01 = np.clip(x / 255.0, 0.0, 1.0)

        # 1) Gamma darkening: x^gamma, gamma > 1 darkens mid-tones
        gamma = random.uniform(*self.gamma_range)
        x_01 = np.power(x_01 + 1e-7, gamma)

        # 2) Brightness scaling: overall dimming
        scale = random.uniform(*self.brightness_range)
        x_01 = x_01 * scale

        # 3) Contrast reduction: blend toward per-channel mean
        contrast = random.uniform(*self.contrast_range)
        if x_01.ndim == 3:       # [C, H, W]
            ch_mean = x_01.mean(axis=(1, 2), keepdims=True)
        elif x_01.ndim == 4:     # [T, C, H, W]
            ch_mean = x_01.mean(axis=(0, 2, 3), keepdims=True)
        else:                    # [B, T, C, H, W]
            ch_mean = x_01.mean(axis=(0, 1, 3, 4), keepdims=True)
        x_01 = contrast * x_01 + (1.0 - contrast) * ch_mean

        # 4) Color temperature shift (simulates artificial lighting)
        if self.channel_num >= 3 and random.random() < self.color_temp_prob:
            r_scale = random.uniform(*self.color_temp_range)
            b_scale = 2.0 - r_scale    # warm R ↔ cool B
            if x_01.ndim == 3:          # [C, H, W]
                x_01[0] *= r_scale
                x_01[2] *= b_scale
            elif x_01.ndim == 4:        # [T, C, H, W]
                x_01[:, 0] *= r_scale
                x_01[:, 2] *= b_scale
            else:                       # [B, T, C, H, W]
                x_01[:, :, 0] *= r_scale
                x_01[:, :, 2] *= b_scale

        # 5) Sensor noise
        noise_scale = self.noise_std / 255.0
        noise = np.random.normal(0.0, noise_scale, x_01.shape).astype(np.float32)
        x_01 = np.clip(x_01 + noise, 0.0, 1.0)

        return (x_01 * 255.0).astype(np.float32)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = self._to_channel_first(x)

        # --- stochastic night simulation (training augmentation) ---
        if random.random() < self.night_prob:
            x = self._apply_night_sim(x)

        # # --- standard ImageNet normalization (identical to parent) ---
        # if x.ndim == 3:
        #     mean = self.mean.reshape((self.channel_num, 1, 1))
        #     std  = self.std.reshape((self.channel_num, 1, 1))
        # elif x.ndim == 4:
        #     mean = self.mean.reshape((1, self.channel_num, 1, 1))
        #     std  = self.std.reshape((1, self.channel_num, 1, 1))
        # elif x.ndim == 5:
        #     mean = self.mean.reshape((1, 1, self.channel_num, 1, 1))
        #     std  = self.std.reshape((1, 1, self.channel_num, 1, 1))
        # else:
        #     raise ValueError(
        #         f'Unsupported shape after layout conversion: {x.shape}')

        # return np.ascontiguousarray((x - mean) / std)
            
        return x


class TextFeatureTransform():
    """Transform per-frame text feature pkl into float32 array.

    Expected pkl content:
      - list length T, each item shape [D], or
      - numpy array [T,D], or
      - numpy array [D].
    """
    def __init__(self, l2_norm=False):
        self.l2_norm = l2_norm

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if self.l2_norm:
            denom = np.linalg.norm(x, axis=-1, keepdims=True)
            x = x / np.maximum(denom, 1e-12)
        return x


class NuisanceMaskTransform():
    """Transform nuisance mask pkl into float32 [T,1] array."""
    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

# gpfgait
# -*- coding: utf-8 -*-
"""
Patch for opengait/data/transform.py.

Append these classes to opengait/data/transform.py, or import them from a
separate file if your OpenGait fork supports external transforms.
"""

import numpy as np


class BaseGPFTransform(object):
    """Keep GPF frame/channel layout and cast to float32.

    Input frame may be:
        [C, H, W] from 000-gpf.pkl
        [H, W] for debugging
    """
    def __call__(self, x):
        return np.asarray(x).astype("float32")


class BaseSilKeepTransform(object):
    """Keep 001-sils.pkl layout.

    Do not use OpenGait's default silhouette cutting transform here, because
    pretreatment_gpf.py has already aligned the silhouette to 64x32.
    """
    def __call__(self, x):
        arr = np.asarray(x)
        if arr.dtype == np.uint8:
            arr = (arr > 0).astype("float32")
        else:
            arr = arr.astype("float32")
        return arr


class BasePhaseTransform(object):
    """Keep compact phase vector [2] as float32."""
    def __call__(self, x):
        return np.asarray(x).astype("float32")


class BasePartMapTransform(object):
    """Keep part maps from 003-parts.pkl as float32."""
    def __call__(self, x):
        return np.asarray(x).astype("float32")
    

class BaseQualityTransform():
    def __call__(self, x):
        return x.astype('float32')


# **************** For RealKineGait ****************
class BaseFeatureTransform():
    """Return precomputed per-frame features, e.g. CLIP text/SMPL teacher embeddings."""
    def __init__(self, dtype='float32', flatten=False):
        self.dtype = dtype
        self.flatten = flatten

    def __call__(self, x):
        x = np.asarray(x).astype(self.dtype)
        if self.flatten:
            x = x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(-1)
        return x


class PoseSequenceTransform():
    """Normalize 2D/3D pose sequences while preserving confidence channel."""
    def __init__(self, center_joint=0, scale_by_bbox=True, dtype='float32'):
        self.center_joint = center_joint
        self.scale_by_bbox = scale_by_bbox
        self.dtype = dtype

    def __call__(self, x):
        x = np.asarray(x).astype(self.dtype)
        if x.ndim < 2:
            return x
        y = x.copy()
        coords = y[..., :2]
        valid = np.isfinite(coords).all(axis=-1, keepdims=True)
        coords = np.where(valid, coords, 0)
        if coords.ndim == 3:
            center = coords[:, self.center_joint:self.center_joint + 1, :]
            coords = coords - center
            if self.scale_by_bbox:
                span = coords.max(axis=1, keepdims=True) - coords.min(axis=1, keepdims=True)
                scale = np.maximum(span.max(axis=-1, keepdims=True), 1.0)
                coords = coords / scale
            y[..., :2] = coords
        return y


class BaseEventVoxelTransform():
    """Normalize event voxel/image tensors to roughly [-1, 1]."""
    def __init__(self, clip_value=10.0, dtype='float32'):
        self.clip_value = clip_value
        self.dtype = dtype

    def __call__(self, x):
        x = np.asarray(x).astype(self.dtype)
        x = np.clip(x, -self.clip_value, self.clip_value) / max(self.clip_value, 1e-6)
        return x
    

"""RealKineGait++ augmentation transforms.

New transforms for causal counterfactual training:
  - SilhouetteContourPerturbation:  dilate torso, simulate coat/backpack/skirt
  - ForegroundColorJitter:           sil-guided colour perturbation (FG only)
  - BackgroundReplacement:           replace background with random noise/color
  - SilhouetteGuidedRGBMask:         zero out BG in RGB (hard version)
  - CoordinatedFlip:                 synchronised flip for RGB + sil
"""

import numpy as np
import random
import cv2
import math


# ===================================================================
# Silhouette Contour Perturbation (Innovation 2 support)
# ===================================================================

class SilhouetteContourPerturbation:
    """Simulate clothing shape changes by morphologically perturbing the sil.

    Operations (applied randomly each call):
      1. Global dilation/erosion — simulates bulk clothing change.
      2. Regional torso dilation — simulates coat/jacket.
      3. Lower-body vertical extension — simulates long skirt/dress.
      4. Upper-body blob — simulates backpack/bag.

    Input:   sil  [S, H, W]  uint8 or float in {0..255}
    Output:  sil  [S, H, W]  same dtype, morphologically perturbed.
    """

    def __init__(self, prob=0.5, max_dilate=5, max_erode=3,
                 torso_prob=0.3, skirt_prob=0.2, backpack_prob=0.2):
        self.prob = prob
        self.max_dilate = max_dilate
        self.max_erode = max_erode
        self.torso_prob = torso_prob
        self.skirt_prob = skirt_prob
        self.backpack_prob = backpack_prob

    def __call__(self, sil):
        if random.random() > self.prob:
            return sil

        was_float = False
        if sil.dtype in (np.float32, np.float64):
            was_float = True
            sil = (sil * 255).astype(np.uint8)

        s, h, w = sil.shape
        result = sil.copy()

        for i in range(s):
            frame = result[i]

            # 1. Global dilation or erosion
            if random.random() < 0.5:
                k = random.randint(1, self.max_dilate)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
                frame = cv2.dilate(frame, kernel, iterations=1)
            elif random.random() < 0.3:
                k = random.randint(1, self.max_erode)
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
                frame = cv2.erode(frame, kernel, iterations=1)

            # 2. Torso dilation (middle 30-60% height range)
            if random.random() < self.torso_prob:
                top = int(h * random.uniform(0.25, 0.35))
                bot = int(h * random.uniform(0.55, 0.70))
                k = random.randint(2, max(3, self.max_dilate))
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
                region = frame[top:bot, :].copy()
                region = cv2.dilate(region, kernel, iterations=1)
                frame[top:bot, :] = region

            # 3. Skirt / long coat extension (extend lower body downward)
            if random.random() < self.skirt_prob:
                # Find the lowest row with foreground
                rows_with_fg = np.where(frame.max(axis=1) > 127)[0]
                if len(rows_with_fg) > 0:
                    last_fg = rows_with_fg[-1]
                    ext = random.randint(3, min(15, h - last_fg - 1))
                    # upper = min(15, h - last_fg - 1)
                    # if upper < 3:
                    #     return img  # 底部空间不足，跳过本次扰动
                    # ext = random.randint(3, upper)

                    if last_fg + ext < h and last_fg > 0:
                        # Copy the last FG row pattern downward with taper
                        for dr in range(1, ext + 1):
                            row = last_fg + dr
                            if row < h:
                                alpha = 1.0 - dr / (ext + 1)
                                frame[row] = np.maximum(
                                    frame[row],
                                    (frame[last_fg].astype(np.float32)
                                     * alpha).astype(np.uint8))

            # 4. Backpack blob (upper-back region)
            if random.random() < self.backpack_prob:
                rows_with_fg = np.where(frame.max(axis=1) > 127)[0]
                if len(rows_with_fg) > 5:
                    top_fg = rows_with_fg[0]
                    mid = top_fg + int(0.3 * (rows_with_fg[-1] - top_fg))
                    cols = np.where(frame[mid] > 127)[0]
                    if len(cols) > 2:
                        cx = int(cols.mean())
                        # Shift slightly to one side (backpack)
                        side = random.choice([-1, 1])
                        cx = cx + side * random.randint(2, max(3, w // 8))
                        cy = mid
                        rx = random.randint(3, max(4, w // 6))
                        ry = random.randint(5, max(6, h // 6))
                        cv2.ellipse(
                            frame, (cx, cy), (rx, ry), 0, 0, 360,
                            255, -1)

            result[i] = frame

        if was_float:
            result = result.astype(np.float32) / 255.0

        return result


# ===================================================================
# Foreground Colour Jitter (Causal counterfactual for clothing colour)
# ===================================================================

class ForegroundColorJitter:
    """Apply colour jitter *only* to the foreground region using silhouette.

    This simulates do(clothing_color = changed) while keeping background
    and body shape intact.

    Expects a dict input: {'rgb': [S,C,H,W], 'sil': [S,H,W]}
    Returns:  dict with modified rgb.
    """

    def __init__(self, prob=0.5,
                 brightness=0.3, contrast=0.3,
                 saturation=0.3, hue=0.1):
        self.prob = prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, data):
        if not isinstance(data, dict) or 'rgb' not in data:
            return data
        if random.random() > self.prob:
            return data

        rgb = data['rgb'].copy()   # [S, C, H, W] or [S, H, W, C]
        sil = data.get('sil', None)
        if sil is None:
            return data

        s = rgb.shape[0]
        # Apply same jitter to all frames (consistent clothing change)
        b_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        c_factor = 1.0 + random.uniform(-self.contrast, self.contrast)

        for i in range(s):
            if rgb.ndim == 4 and rgb.shape[1] == 3:
                # [S, C, H, W]
                frame = rgb[i].transpose(1, 2, 0)  # → [H, W, C]
            else:
                frame = rgb[i]

            mask = (sil[i] > 0.5 if sil.max() <= 1.0
                    else sil[i] > 127).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

            jittered = frame.copy().astype(np.float32)
            # Brightness
            jittered = jittered * b_factor
            # Contrast
            mean = jittered.mean()
            jittered = (jittered - mean) * c_factor + mean
            jittered = np.clip(jittered, 0, 255)

            # Blend: FG gets jittered, BG stays original
            frame_out = frame * (1 - mask) + jittered * mask

            if rgb.ndim == 4 and rgb.shape[1] == 3:
                rgb[i] = frame_out.transpose(2, 0, 1)
            else:
                rgb[i] = frame_out

        data['rgb'] = rgb
        return data


# ===================================================================
# Background Replacement
# ===================================================================

class BackgroundReplacement:
    """Replace background with random solid colour or noise.

    Simulates do(background = changed).
    Expects dict input: {'rgb': [S,C,H,W], 'sil': [S,H,W]}.
    """

    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, data):
        if not isinstance(data, dict) or 'rgb' not in data:
            return data
        if random.random() > self.prob:
            return data

        rgb = data['rgb'].copy()
        sil = data.get('sil', None)
        if sil is None:
            return data

        s = rgb.shape[0]
        # Choose a strategy
        strategy = random.choice(['solid', 'noise', 'blur'])

        if strategy == 'solid':
            bg_color = np.array([random.randint(0, 255) for _ in range(3)],
                                dtype=np.float32)
        elif strategy == 'noise':
            bg_color = None
        else:  # blur
            bg_color = 'blur'

        for i in range(s):
            if rgb.ndim == 4 and rgb.shape[1] == 3:
                frame = rgb[i].transpose(1, 2, 0).copy()
            else:
                frame = rgb[i].copy()

            mask = (sil[i] > 0.5 if sil.max() <= 1.0
                    else sil[i] > 127).astype(np.float32)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]

            h, w = frame.shape[:2]
            if bg_color is None:
                bg = np.random.randint(0, 256, (h, w, 3)).astype(np.float32)
            elif bg_color == 'blur':
                bg = cv2.GaussianBlur(
                    frame.astype(np.float32), (31, 31), 15)
            else:
                bg = np.full((h, w, 3), bg_color, dtype=np.float32)

            frame_out = frame * mask + bg * (1 - mask)

            if rgb.ndim == 4 and rgb.shape[1] == 3:
                rgb[i] = frame_out.transpose(2, 0, 1)
            else:
                rgb[i] = frame_out

        data['rgb'] = rgb
        return data


# ===================================================================
# Hard Foreground Mask (sil × RGB, zero out BG)
# ===================================================================

class SilhouetteGuidedRGBMask:
    """Zero out background pixels in RGB using silhouette.

    Output: rgb_fg = rgb * sil (per frame).
    Expects dict or separate arrays.
    """

    def __init__(self, apply_prob=1.0):
        self.apply_prob = apply_prob

    def __call__(self, data):
        if not isinstance(data, dict) or 'rgb' not in data:
            return data
        if random.random() > self.apply_prob:
            return data

        rgb = data['rgb'].copy()
        sil = data.get('sil', None)
        if sil is None:
            return data

        s = rgb.shape[0]
        for i in range(s):
            mask = (sil[i] > 0.5 if sil.max() <= 1.0
                    else sil[i] > 127).astype(np.float32)
            if rgb.ndim == 4 and rgb.shape[1] == 3:
                # [C, H, W]
                rgb[i] = rgb[i] * mask[np.newaxis]
            else:
                if mask.ndim == 2:
                    mask = mask[:, :, np.newaxis]
                rgb[i] = rgb[i] * mask

        data['rgb'] = rgb
        return data


# ===================================================================
# Coordinated Flip (sync RGB + sil + pose)
# ===================================================================

class CoordinatedFlip:
    """Horizontally flip RGB, silhouette, and pose together.

    Expects dict: {'rgb': ..., 'sil': ..., 'pose': ...}
    """

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, data):
        if not isinstance(data, dict):
            return data
        if random.random() > self.prob:
            return data

        if 'rgb' in data:
            data['rgb'] = data['rgb'][..., ::-1].copy()
        if 'sil' in data:
            data['sil'] = data['sil'][..., ::-1].copy()
        if 'pose' in data:
            pose = data['pose'].copy()
            # Flip x-coordinates (assume first coordinate is x)
            if pose.ndim >= 3:
                # [T, J, C] — flip x
                w = 1.0  # Normalised coordinates: flip around 0
                pose[..., 0] = -pose[..., 0]
            data['pose'] = pose

        return data

# **************** For RealKineGait++ v2 ****************

class BaseFeatureTransform():
    """Pass-through transform for per-frame feature vectors.

    It is used for motion_text, nuisance_text, SMPL vectors, and other
    precomputed descriptors saved in OpenGait sequence folders.  The return
    value is always float32 so BaseModel.inputs_pretreament can safely convert
    it to a torch tensor.
    """

    def __init__(self, squeeze=True):
        self.squeeze = squeeze

    def __call__(self, x):
        arr = np.asarray(x, dtype=np.float32)
        if self.squeeze:
            arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr


class PoseSequenceTransform():
    """Normalize pose sequences to the model's expected [T, J, 3] layout.

    Accepted inputs:
    - [T, J, 3] or [T, J, 2]
    - [T, J*3] or [T, J*2]

    Coordinates are converted to [-1, 1].  If ``image_size=(H, W)`` is given,
    pixel coordinates are normalized by W/H.  Otherwise the transform handles
    common [0,1] normalized inputs and falls back to sequence-wise max-value
    scaling for pixel-like coordinates.
    """

    def __init__(self, num_joints=17, image_size=None, normalize=True,
                 confidence_fill=1.0, min_valid_conf=0.0):
        self.num_joints = int(num_joints)
        self.image_size = image_size
        self.normalize = normalize
        self.confidence_fill = float(confidence_fill)
        self.min_valid_conf = float(min_valid_conf)

    def _to_tjc(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            if arr.size % 3 == 0:
                arr = arr.reshape(1, arr.size // 3, 3)
            elif arr.size % 2 == 0:
                arr = arr.reshape(1, arr.size // 2, 2)
        elif arr.ndim == 2:
            if arr.shape[-1] in (2, 3):
                arr = arr[None, ...]
            elif arr.shape[-1] % 3 == 0:
                arr = arr.reshape(arr.shape[0], arr.shape[-1] // 3, 3)
            elif arr.shape[-1] % 2 == 0:
                arr = arr.reshape(arr.shape[0], arr.shape[-1] // 2, 2)
        if arr.ndim != 3:
            raise ValueError(f'PoseSequenceTransform expects [T,J,C] or [T,J*C], got shape {arr.shape}.')
        if arr.shape[-1] == 2:
            conf = np.full(arr.shape[:-1] + (1,), self.confidence_fill, dtype=np.float32)
            arr = np.concatenate([arr, conf], axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]
        return arr.astype(np.float32, copy=False)

    def _normalize_xy(self, arr):
        xy = arr[..., :2]
        finite = np.isfinite(xy)
        xy = np.where(finite, xy, 0.0)
        if self.image_size is not None:
            h, w = self.image_size
            scale = np.array([max(float(w) - 1.0, 1.0), max(float(h) - 1.0, 1.0)], dtype=np.float32)
            xy = xy / scale * 2.0 - 1.0
        else:
            xy_min = float(np.min(xy)) if xy.size else 0.0
            xy_max = float(np.max(xy)) if xy.size else 0.0
            if xy_min >= 0.0 and xy_max <= 1.5:
                xy = xy * 2.0 - 1.0
            elif xy_min < -1.5 or xy_max > 1.5:
                # Pixel-like coordinates without a known image size.  Estimate
                # separate x/y scales from the current sequence; keep empty
                # joints near zero rather than pushing them to -1.
                valid = np.abs(xy).sum(-1) > 1e-6
                if valid.any():
                    max_x = max(float(np.max(xy[..., 0][valid])), 1.0)
                    max_y = max(float(np.max(xy[..., 1][valid])), 1.0)
                    xy[..., 0] = xy[..., 0] / max_x * 2.0 - 1.0
                    xy[..., 1] = xy[..., 1] / max_y * 2.0 - 1.0
                    xy[~valid] = 0.0
        arr[..., :2] = np.clip(xy, -1.0, 1.0)
        arr[..., 2] = np.clip(np.nan_to_num(arr[..., 2], nan=0.0), 0.0, 1.0)
        if self.min_valid_conf > 0:
            arr[..., 2] = np.where(arr[..., 2] >= self.min_valid_conf, arr[..., 2], 0.0)
        return arr

    def __call__(self, x):
        arr = self._to_tjc(x)
        if arr.shape[1] < self.num_joints:
            pad = np.zeros((arr.shape[0], self.num_joints - arr.shape[1], 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self.num_joints:
            arr = arr[:, :self.num_joints]
        if self.normalize:
            arr = self._normalize_xy(arr)
        return arr.astype(np.float32, copy=False)


# class SilhouetteContourPerturbation():
#     """Lightweight silhouette morphology augmentation.

#     The class is intentionally dependency-light and works with [T,H,W] and
#     [T,1,H,W] arrays.  It randomly dilates or erodes each frame, preserving the
#     original shape and float range.
#     """

#     def __init__(self, prob=0.3, max_dilate=4, max_erode=None):
#         self.prob = float(prob)
#         self.max_dilate = int(max_dilate)
#         self.max_erode = int(max_erode) if max_erode is not None else int(max_dilate)

#     def __call__(self, seq):
#         if random.uniform(0, 1) >= self.prob:
#             return seq
#         arr = np.asarray(seq)
#         orig_shape = arr.shape
#         has_channel = arr.ndim == 4
#         if has_channel:
#             frames = arr[:, 0]
#         else:
#             frames = arr
#         out = []
#         op_is_dilate = random.random() < 0.5
#         kmax = self.max_dilate if op_is_dilate else self.max_erode
#         k = max(1, random.randint(1, max(1, kmax)))
#         kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
#         for f in frames:
#             dtype = f.dtype
#             mx = float(np.max(f)) if f.size else 1.0
#             img = (f > (0.5 if mx <= 1.5 else 127.5)).astype(np.uint8) * 255
#             img = cv2.dilate(img, kernel, iterations=1) if op_is_dilate else cv2.erode(img, kernel, iterations=1)
#             img = img.astype(np.float32) / 255.0
#             if mx > 1.5:
#                 img = img * 255.0
#             out.append(img.astype(dtype, copy=False))
#         out = np.stack(out, axis=0)
#         if has_channel:
#             out = out[:, None]
#         return out.reshape(orig_shape)
