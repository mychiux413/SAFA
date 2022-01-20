from io import BytesIO
import os
import cv2
from skimage import io, img_as_float32
from skimage.color import gray2rgb
from sklearn.model_selection import train_test_split
from imageio import mimread

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from augmentation import AllAugmentationTransform
import glob
import json
import pickle


def read_video(name, frame_shape):
    """
    Read video which can be:
      - an image of concatenated frames
      - '.mp4' and'.gif'
      - folder with videos
    """

    if os.path.isdir(name):
        frames = sorted(os.listdir(name))
        num_frames = len(frames)
        video_array = np.array(
            [img_as_float32(io.imread(os.path.join(name, frames[idx]))) for idx in range(num_frames)])
    elif name.lower().endswith('.png') or name.lower().endswith('.jpg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)

        video_array = np.moveaxis(image, 1, 0)

        video_array = video_array.reshape((-1,) + frame_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name))
        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception("Unknown file extensions  %s" % name)

    return video_array


class FramesDataset(Dataset):
    """
    Dataset of videos, each video can be represented as:
      - an image of concatenated frames
      - '.mp4' or '.gif'
      - folder with all frames
    """

    def __init__(self, root_dir, meta_dir, frame_shape=(256, 256, 3), id_sampling=False, is_train=True,
                 random_seed=0, pairs_list=None, augmentation_params=None):
        self.root_dir = root_dir
        self.videos = os.listdir(root_dir)
        self.frame_shape = tuple(frame_shape)
        self.pairs_list = pairs_list
        self.id_sampling = id_sampling
        if os.path.exists(os.path.join(root_dir, 'train')):
            assert os.path.exists(os.path.join(root_dir, 'test'))
            print("Use predefined train-test split.")
            if id_sampling:
                train_videos = {os.path.basename(video).split('#')[0] for video in
                                os.listdir(os.path.join(root_dir, 'train'))}
                train_videos = list(train_videos)
            else:
                train_videos = os.listdir(os.path.join(root_dir, 'train'))
            test_videos = os.listdir(os.path.join(root_dir, 'test'))
            self.root_dir = os.path.join(self.root_dir, 'train' if is_train else 'test')
        else:
            print("Use random train-test split.")
            train_videos, test_videos = train_test_split(self.videos, random_state=random_seed, test_size=0.2)

        if is_train:
            self.videos = train_videos
        else:
            self.videos = test_videos

        self.is_train = is_train

        if self.is_train and augmentation_params is not None:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

        self.meta_dir = meta_dir

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        if self.is_train and self.id_sampling:
            name = self.videos[idx]
            path = np.random.choice(glob.glob(os.path.join(self.root_dir, name + '*.mp4')))
        else:
            name = self.videos[idx]
            path = os.path.join(self.root_dir, name)

        video_name = os.path.basename(path)

        if self.is_train and os.path.isdir(path):
            frames = os.listdir(path)
            num_frames = len(frames)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2))
            video_array = [img_as_float32(io.imread(os.path.join(path, frames[idx]))) for idx in frame_idx]
        else:
            video_array = read_video(path, frame_shape=self.frame_shape)
            num_frames = len(video_array)
            frame_idx = np.sort(np.random.choice(num_frames, replace=True, size=2)) if self.is_train else range(
                num_frames)
            video_array = video_array[frame_idx]

        if self.transform is not None:
            video_array, t_flip, h_flip = self.transform(video_array)
            if t_flip:
                frame_idx = frame_idx[::-1]

        out = {}
        if self.is_train:
            f = open(os.path.join(self.meta_dir, video_name.split('.')[0] + '.pkl'), 'rb')
            video_meta = pickle.load(f)
            f.close()
            source = np.array(video_array[0], dtype='float32')
            driving = np.array(video_array[1], dtype='float32')

            out['driving'] = driving.transpose((2, 0, 1))
            out['source'] = source.transpose((2, 0, 1))
            out['source_ldmk_2d'] = video_meta[frame_idx[0]]['ldmk']
            out['driving_ldmk_2d'] = video_meta[frame_idx[1]]['ldmk']
            
            if h_flip:
                out['source_ldmk_2d'][:, 0] = source.shape[1] - out['source_ldmk_2d'][:, 0]
                out['driving_ldmk_2d'][:, 0] = driving.shape[1] - out['driving_ldmk_2d'][:, 0]
        else:
            video = np.array(video_array, dtype='float32')
            out['video'] = video.transpose((3, 0, 1, 2))

        out['name'] = video_name

        return out


class PairedDataset(Dataset):
    """
    Dataset of pairs for animation.
    """

    def __init__(self, initial_dataset, number_of_pairs, seed=0):
        self.initial_dataset = initial_dataset
        pairs_list = self.initial_dataset.pairs_list

        np.random.seed(seed)

        if pairs_list is None:
            max_idx = min(number_of_pairs, len(initial_dataset))
            nx, ny = max_idx, max_idx
            xy = np.mgrid[:nx, :ny].reshape(2, -1).T
            number_of_pairs = min(xy.shape[0], number_of_pairs)
            self.pairs = xy.take(np.random.choice(xy.shape[0], number_of_pairs, replace=False), axis=0)
        else:
            videos = self.initial_dataset.videos
            name_to_index = {name: index for index, name in enumerate(videos)}
            pairs = pd.read_csv(pairs_list)
            pairs = pairs[np.logical_and(pairs['source'].isin(videos), pairs['driving'].isin(videos))]

            number_of_pairs = min(pairs.shape[0], number_of_pairs)
            self.pairs = []
            self.start_frames = []
            for ind in range(number_of_pairs):
                self.pairs.append(
                    (name_to_index[pairs['driving'].iloc[ind]], name_to_index[pairs['source'].iloc[ind]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        first = self.initial_dataset[pair[0]]
        second = self.initial_dataset[pair[1]]
        first = {'driving_' + key: value for key, value in first.items()}
        second = {'source_' + key: value for key, value in second.items()}

        return {**first, **second}


class DatasetRepeater(Dataset):
    """
    Pass several times over the same dataset for better i/o performance
    """

    def __init__(self, dataset, num_repeats=100):
        self.dataset = dataset
        self.num_repeats = num_repeats

    def __len__(self):
        return self.num_repeats * self.dataset.__len__()

    def __getitem__(self, idx):
        return self.dataset[idx % self.dataset.__len__()]


class ImageDataset(Dataset):
    '''
    Dataset of images for the pre-training of tdmm estimator
    '''
    def __init__(self, data_dir, meta_dir=None, augmentation_params=None):
        if augmentation_params:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

        self.data_list = glob.glob(os.path.join(data_dir, '*', '*.png'))

        self.meta_dir = meta_dir

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_pth = self.data_list[idx]
        frame_idx = int(os.path.basename(image_pth).split('.')[0])
        video_name = image_pth.split('/')[-2]

        image = [img_as_float32(io.imread(image_pth))]

        if self.transform:
            image, _, f_flip = self.transform(image)
        
        image = np.array(image[0], dtype='float32')

        out = {}
        out['image'] = image.transpose((2, 0, 1))

        if self.meta_dir is not None:
            f = open(os.path.join(self.meta_dir, video_name.split('.')[0] + '.pkl'), 'rb')
            video_meta = pickle.load(f)
            f.close()
            out['ldmk'] = video_meta[frame_idx]['ldmk']

            if f_flip:
                out['ldmk'][:, 0] = image.shape[1] - out['ldmk'][:, 0]

        return out


class DBImageDataset(object):

    def __init__(
            self, dataset_type, data_root, share_root,
            db_address='localhost', db_name='speech',
            limit=0, augmentation_params=None,
            ):
        if augmentation_params:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

        assert os.path.exists(share_root)
        self.db_address = db_address
        self.db_name = db_name

        self.conn = None
        self.init_connection()
        script = """
            SELECT y.id, y.frame_count FROM youtube_speech AS y
            JOIN dataset_type AS d
                ON d.id = y.dataset_type_id
            WHERE
                d.value {}
                AND y.valid = true
                AND y.path IS NOT NULL
                AND y.mel IS NOT NULL
        """
        print(f"**** Dataset Type: {dataset_type} ****")
        if dataset_type is None:
            script = script.format(" IS NOT NULL")
        elif isinstance(dataset_type, list):
            dataset_type = " IN ({})".format(", ".join([f"'{d}'" for d in dataset_type]))
        elif dataset_type == "train":
            dataset_type = "NOT LIKE '%test%'"
        elif dataset_type == "test":
            dataset_type = "LIKE '%test%'"
        else:
            dataset_type = f"= '{dataset_type}'"
        script = script.format(dataset_type)

        if limit > 0:
            script += "\nlimit {}".format(limit)
        self.cur.execute(script)

        self.landmarks_cache = {}
        self.data_ids = []
        for row in self.cur.fetchall():
            self.landmarks_cache[row['id']] = None
            self.data_ids.append((row['id'], row['frame_count']))
        self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.share_root = share_root
        self.data_len = len(self.data_ids)

        print(f"{self.data_len} videos")
        assert self.data_len > 0, f'no data found, the script is: {script}'
        self.conn.close()
        self.close_connection()

        self.info_script = """
        SELECT bboxes, landmarks, frame_count, path, h, w FROM youtube_speech
        WHERE id = {}
        """

    def init_connection(self):
        if self.conn is not None:
            return
        from mysql.connector import connect
        self.conn = connect(
            user="root",
            password="root",
            host=self.db_address,
            port="3456",
            database=self.db_name,
            charset='utf8mb4',
        )
        self.cur = self.conn.cursor(dictionary=True)

    def close_connection(self):
        self.cur.close()
        self.conn.close()
        self.conn = None

    def get_video_info(self, youtube_speech_id: int):
        self.init_connection()
        self.cur.execute(self.info_script.format(youtube_speech_id))
        row = self.cur.fetchone()
        bboxes = np.load(BytesIO(row['bboxes']))
        landmarks = np.load(BytesIO(row['landmarks'])).astype(np.float32)
        return {
            'bboxes': bboxes,
            'landmarks': landmarks,
            'path': row['path'],
            'frame_count': row['frame_count'],
            'h': row['h'],
            'w': row['w'],
        }

    def read_and_cache(self, youtube_speech_id, iframe):
        img_path = os.path.join(self.data_root, str(youtube_speech_id), f'{iframe}.png')
        if not os.path.exists(img_path):
            os.makedirs(os.path.join(self.data_root, str(youtube_speech_id)), exist_ok=False)
            info = self.get_video_info(youtube_speech_id)
            src_video_path = os.path.join(self.share_root, info['path'])
            assert os.path.exists(src_video_path)


            # *****  each ****
            landmarks = info['landmarks']
            bboxes = info['bboxes']
            for i in range(info['frame_count']):
                bbox = bboxes[i]
                x0, y0, x1, y1 = bbox
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(info['w'], x1)
                y1 = min(info['h'], y1)
                w = x1 - x0
                h = y1 - y0

                bboxes[i, 0] = x0
                bboxes[i, 1] = y0
                bboxes[i, 2] = x1
                bboxes[i, 3] = y1

                landmarks[i, :, 0] = landmarks[i, :, 0] - x0
                landmarks[i, :, 1] = landmarks[i, :, 1] - y0

                landmarks[i, :, 0] = landmarks[i, :, 0] * 256 / w
                landmarks[i, :, 1] = landmarks[i, :, 1] * 256 / h
            # ***********

            # bboxes = info['bboxes']
            # landmarks = info['landmarks']
            # x0 = max(0, bboxes[:, 0].mean())
            # y0 = max(0, bboxes[:, 1].mean())
            # x1 = min(info['w'], bboxes[:, 2].mean())
            # y1 = min(info['h'], bboxes[:, 3].mean())
            # w = x1 - x0
            # h = y1 - y0

            # landmarks[:, :, 0] = landmarks[:, :, 0] - x0
            # landmarks[:, :, 1] = landmarks[:, :, 1] - y0

            # landmarks[:, :, 0] = landmarks[:, :, 0] * 256 / w
            # landmarks[:, :, 1] = landmarks[:, :, 1] * 256 / h

            landmarks_path = os.path.join(self.data_root, str(youtube_speech_id), 'landmarks.npy')
            np.save(landmarks_path, landmarks)
            self.landmarks_cache[youtube_speech_id] = landmarks

            cap = cv2.VideoCapture(src_video_path)
            for i in range(info['frame_count']):
                _, img = cap.read()

                bbox = bboxes[i]
                x0, y0, x1, y1 = bbox
                img = img[y0:y1, x0:x1]
                img = cv2.resize(img, (256, 256))
                path = os.path.join(self.data_root, str(youtube_speech_id), f'{i}.png')
                cv2.imwrite(path, img)
                if i == iframe:
                    output_img = img
            cap.release()
        else:
            output_img = cv2.imread(img_path)
        output_img = output_img[:, :, ::-1]
        output_img = output_img.astype(np.float32) / 255.0
        landmarks = self.read_landmarks(youtube_speech_id)
        ldmk = landmarks[iframe]
        return {
            'image': output_img,
            'ldmk': ldmk,
        }

    def read_landmarks(self, youtube_speech_id: int):
        """return: frame_count x 68 x 2"""
        landmarks = self.landmarks_cache.get(youtube_speech_id)
        if landmarks is not None:
            return landmarks
        path = os.path.join(self.data_root, str(youtube_speech_id), 'landmarks.npy')
        landmarks = np.load(path)
        self.landmarks_cache[youtube_speech_id] = landmarks
        return landmarks

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        youtube_speech_id, frame_count = self.data_ids[idx]
        iframe = np.random.randint(frame_count)
        out = self.read_and_cache(youtube_speech_id, iframe)

        image = [out['image']]
        f_flip = False
        if self.transform:
            image, _, f_flip = self.transform(image)
        
        image = np.array(image[0], dtype='float32')

        out['image'] = image.transpose((2, 0, 1))
        if f_flip:
            out['ldmk'][:, 0] = image.shape[1] - out['ldmk'][:, 0]
        return out


class DBImageDataset2(object):

    def __init__(
            self, dataset_type, data_root, share_root,
            db_address='localhost', db_name='speech',
            limit=0, augmentation_params=None,
            ):
        if augmentation_params:
            self.transform = AllAugmentationTransform(**augmentation_params)
        else:
            self.transform = None

        assert os.path.exists(share_root)
        self.db_address = db_address
        self.db_name = db_name

        self.init_connection()
        script = """
            SELECT y.id, y.path, y.frame_count FROM youtube_speech AS y
            JOIN dataset_type AS d
                ON d.id = y.dataset_type_id
            WHERE
                d.value {}
                AND y.valid = true
                AND y.path IS NOT NULL
                AND y.mel IS NOT NULL
        """
        print(f"**** Dataset Type: {dataset_type} ****")
        if dataset_type is None:
            script = script.format(" IS NOT NULL")
        elif isinstance(dataset_type, list):
            dataset_type = " IN ({})".format(", ".join([f"'{d}'" for d in dataset_type]))
        elif dataset_type == "train":
            dataset_type = "NOT LIKE '%test%'"
        elif dataset_type == "test":
            dataset_type = "LIKE '%test%'"
        else:
            dataset_type = f"= '{dataset_type}'"
        script = script.format(dataset_type)

        if limit > 0:
            script += "\nlimit {}".format(limit)
        self.cur.execute(script)

        self.landmarks_cache = {}
        self.id_path_map = {}
        self.data_ids = []
        for row in self.cur.fetchall():
            self.id_path_map[row['id']] = row['path']
            self.landmarks_cache[row['id']] = None
            for j in range(row['frame_count']):
                self.data_ids.append((row['id'], j))
        self.data_root = data_root
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        self.share_root = share_root
        self.data_len = len(self.data_ids)

        print(f"{len(self.id_path_map)} videos, {self.data_len} data length")
        assert self.data_len > 0, f'no data found, the script is: {script}'
        self.conn.close()
        self.close_connection()

    def init_connection(self):
        from mysql.connector import connect
        self.conn = connect(
            user="root",
            password="root",
            host=self.db_address,
            port="3456",
            database=self.db_name,
            charset='utf8mb4',
        )
        self.cur = self.conn.cursor(dictionary=True)

    def close_connection(self):
        self.cur.close()
        self.conn.close()
        self.conn = None

    def read_landmarks(self, youtube_speech_id: int, dump_dirpath: str):
        """return: frame_count x 68 x 2"""
        landmarks = self.landmarks_cache.get(youtube_speech_id)
        if landmarks is not None:
            return landmarks
        path = os.path.join(dump_dirpath, 'landmarks.npy')
        landmarks = np.load(path) * 255.0
        self.landmarks_cache[youtube_speech_id] = landmarks
        return landmarks
    @classmethod
    def img_as_float32(cls, path):
        img = cv2.imread(path)[:, :, ::-1]
        img = img.astype(np.float32) / 255.0
        return img

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        youtube_speech_id, iframe = self.data_ids[idx]
        path = self.id_path_map[youtube_speech_id]
        filename = os.path.basename(path)
        dump_dirpath = os.path.join(self.data_root, filename[:-4])
        img_path = os.path.join(dump_dirpath, f'{iframe}.webp')

        image = [self.img_as_float32(img_path)]

        f_flip = False
        if self.transform:
            image, _, f_flip = self.transform(image)
        
        image = np.array(image[0], dtype='float32')

        out = {}
        out['image'] = image.transpose((2, 0, 1))
        landmarks = self.read_landmarks(youtube_speech_id, dump_dirpath)
        out['ldmk'] = landmarks[iframe]
        if f_flip:
            print("**** NO!!! FLIP !!! ****")
            out['ldmk'][:, 0] = image.shape[1] - out['ldmk'][:, 0]
        return out
