#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
import pandas as pd

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Kinetics(torch.utils.data.Dataset):
    """
    Kinetics video loader. Construct the Kinetics video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Kinetics video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Kinetics {}...".format(mode))
        self._construct_loader(debug=False)

    def _construct_loader(self, debug=False):
        """
        Construct the video loader.
        """
        print('Kinetics data loader...')
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._center_frames = []
        self._labels = []
        self._spatial_temporal_idx = []
        # Other fields we may want to debug:
        self._info = []

        # Handle as CSV file from pandas instead
        #with open(path_to_file, "r") as f:
        df = pd.read_csv(path_to_file)
        if True:
            print('Reading file %s' % path_to_file)
            #for clip_idx, path_label in enumerate(f.read().splitlines()):
            for clip_idx, data in df.iterrows():
                if debug:
                    print('------------------------')
                # Useful, but too much display
                if debug:
                    print(clip_idx, data)
                #assert len(path_label.split()) == 2
                #path, label = path_label.split()
                #path, label = path_label.split()[:2]
                path, label, center_frame = data['filepath'], data['pitchType'], data['ReleaseFrame']
                if debug:
                    print('Clip %d' % clip_idx)
                    print((path, label, center_frame))
                # TODO: Table convert text to emum
                label = utils.PITCH_TYPE_ENUM[label]

                # Hack -- use regexp to skip hard players -- lefties?
                # TODO: If we want to measure validation on pitcher unseen by training
                #if path.find("McSteen_J") != -1:
                #    print('skipping LEFTY %s' % path)
                #    continue

                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    # HACK -- center frame if video -- if provided
                    self._center_frames.append(float(center_frame))
                    # Save all the info (for debug purpose)
                    self._info.append({k:data[k] for k in data.keys()})
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Kinetics split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index, hack_center_frame=200., debug=False):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        # TODO: Data augmentation? In training only. Flip image ~1/3 of the time? To deal with lefties.
        # TODO: Apply gaussian blurr -- to the "pitch design + text" tab?
        # TODO: Other paranoia about visualization and accidental labels?

        if debug:
            print('Calling Kinetics __getitem__()')
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if debug:
                print('[temporal_sample_index, spatial_sample_index, min_scale, max_scale, crop_size]')
                print([temporal_sample_index, spatial_sample_index, min_scale, max_scale, crop_size])
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            video_container = None
            try:
                video_container = container.get_video_container(
                    self._path_to_videos[index],
                    self.cfg.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE,
                )
                hack_center_frame = self._center_frames[index]
            except Exception as e:
                logger.info(
                    "Failed to load video from {} with error {}".format(
                        self._path_to_videos[index], e
                    )
                )

            # Select a random video if the current video was not able to access.
            if video_container is None:
                print('<ERROR>: take random video')
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            else:
                if debug:
                    print('Found video!')

            # Decode video. Meta info is used to perform selective decoding.
            #print('decoding frames...')
            frames = decoder.decode(
                video_container,
                self.cfg.DATA.SAMPLING_RATE,
                self.cfg.DATA.NUM_FRAMES,
                temporal_sample_index,
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                video_meta=self._video_meta[index],
                target_fps=30,
                hack_center_frame=hack_center_frame,
                debug=debug,
            )
            if not(frames is None):
                if debug:
                    print('Got frames')
                    print(frames.shape)
            else:
                print('Frames fail!')

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                print('<ERROR>: take random video')
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            if debug:
                print('Shape into spatial sampling: %s' % str(frames.shape))
            frames = self.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                debug=debug,
            )
            if debug:
                print('Shape after spatial sampling: %s' % str(frames.shape))

            label = self._labels[index]
            info = self._info[index]
            # HACK -- collect "extra information" for loss function
            # NOTE -- make sure to include spinAxisDegrees last! Because we use custom loss for that...
            extra_label_cols = ['speed_norm', 'spin_norm', 'trueSpin_norm', 'spinEfficiency_norm',
                'topSpin_norm', 'sideSpin_norm', 'rifleSpin_norm',
                'vb_norm', 'hb_norm', 'hAngle_norm', 'rAngle_norm', 'armAngle_norm', 'spinAxisDeg']
            extra_labels = [info[c] for c in extra_label_cols]
            name_cols = ['filepath', 'spinAxis', 'spinAxisDeg']
            fname = [info[c] for c in name_cols]
            if debug:
                print('extra labels -- speed, spin etc [normalized]')
                print(extra_labels)
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, (label, extra_labels, fname), index, {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        debug=False,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        if debug:
            print('[frames.shape, spatial_idx, min_scale, max_scale, crop_size]')
            print([frames.shape, spatial_idx, min_scale, max_scale, crop_size])
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.random_crop(frames, crop_size)
            frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
