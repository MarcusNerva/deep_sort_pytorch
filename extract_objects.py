import pickle
import subprocess
import numpy as np
import glob
import os
from utils.backbone import get_resnext101_32x8d
import torch
from torchvision import transforms
from torchvision.ops.roi_align import RoIAlign
from PIL import Image


obtain_tracks_cmd = 'python /home/hanhuaye/PythonProject/opensource/deep_sort_pytorch/yolov3_deepsort.py ' \
                    '{video_path} --save_path {save_path}'
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
trans = transforms.Compose([
    transforms.ToTensor(),
    normalize
])
resnext_dim = 2048


def process_pos(pos, width, height):
    pos[0] = max(0, pos[0])
    pos[1] = max(0, pos[1])
    pos[2:] = pos[:2] + pos[2:]

    x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
    x1 = min(x1, width)
    x2 = min(x2, width)
    y1 = min(y1, height)
    y2 = min(y2, height)

    return np.array([x1, y1, x2, y2], dtype=float)


def get_visual_data(save_dir, data_dir, video_name, resnext101, device):
    tracks_path = os.path.join(save_dir, 'tracks.pkl')
    length_path = os.path.join(save_dir, 'length.pkl')
    video_path = os.path.join(data_dir, video_name + '.avi')
    visual_feature_dir = os.path.join(data_dir, 'objects')
    class_feature_dir = os.path.join(data_dir, 'category')
    visual_feature_path = os.path.join(visual_feature_dir, video_name + '.npy')
    class_feature_path = os.path.join(class_feature_dir, video_name + '.npy')
    objects_mask_path = os.path.join(visual_feature_dir, video_name + '_mask.npy')
    hastracks_path = os.path.join(visual_feature_dir, video_name + '_hastracks.pkl')

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    with open(length_path, 'rb') as f:
        frames_numb = pickle.load(f)

    if len(tracks) == 0:
        has_tracks = False
        visual_features = np.zeros([15, 1, 2048, 4, 4])
        objects_mask = np.zeros(15, 1)
        class_features = np.zeros([1, 1000], dtype=int)

        np.save(visual_feature_path, visual_features)
        np.save(class_feature_path, class_features)
        np.save(objects_mask_path, objects_mask)
        with open(hastracks_path, 'wb') as f:
            pickle.dump(has_tracks, f)
        return

    visit = np.zeros(frames_numb + 5, dtype=int)
    frames_store = []

    import cv2
    vdo = cv2.VideoCapture()
    vdo.open(video_path)
    im_width, im_height = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while vdo.grab():
        _, ori_im = vdo.retrieve()
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        frames_store.append(trans(im).unsqueeze(0).to(device))
    assert len(frames_store) == frames_numb, \
        'expect len(frames_store) == frames_numb, ' \
        'but got {} and {}'.format(len(frames_store), frames_numb)

    interval = frames_numb // 15
    tracks = [item for item in tracks if len(item['frame_ids']) >= interval]
    tracks = sorted(tracks, key=lambda x: len(track['frame_ids']), reverse=True)
    if len(tracks) > 20:
        tracks = tracks[:20]
    num_tracks = len(tracks)

    for track in tracks:
        frames_ids = track['frame_ids']
        for id in frames_ids:
            visit[id] = 1

    class_store = []
    for track in tracks:
        class_id = tracks['class_id']
        class_vec = np.zeros(1000, dtype=int)
        class_vec[class_id] = 1
        class_store.append(class_vec)
    class_store = np.concatenate([item.unsqueeze(0) for item in class_store], axis=0)
    assert class_store.shape == (num_tracks, 1000), \
        'expected class_store.shape == ({}, 1000), but got {}'.\
            format(num_tracks, class_store.shape)

    time_span = 15
    step = int(visit.sum()) // time_span
    anchor_idxs = []
    has_tracks = True
    cnt = 0
    for i in range(frames_numb):
        cnt += 1 if visit[i] > 0 else 0
        if visit[i] > 0 and cnt % step == 0:
            anchor_idxs.append(i)
    assert len(anchor_idxs) == time_span, \
        'expected len(anchor_idxs) == time_span, but got {}'.format(len(anchor_idxs))

    pool_size = 4
    spatial_scale = 1.0 / 32.0
    fake_feature = np.zeros([1, resnext_dim, pool_size, pool_size])
    roi_align = RoIAlign((pool_size, pool_size), spatial_scale=spatial_scale, sampling_ratio=1).to(device)

    result_feature = []
    objects_mask = np.zeros([15, len(tracks)])
    for i, idx in enumerate(anchor_idxs):
        temp_store = []
        feature_map = resnext101(frames_store[idx])  # (1, 2048, H/32, W/32)
        for j, item in enumerate(tracks):
            frames_ids, positions, class_id = item['frame_ids'], item['positions'], item['class_id']
            if idx not in frames_ids:
                objects_mask[i, j] = 0
                temp_store.append(fake_feature)
            else:
                ptr = frames_ids.index(idx)
                position = positions[ptr]
                position = process_pos(position, im_width, im_height)
                x1, y1, x2, y2 = position[0], position[1], position[2], position[3]
                bbox = torch.FloatTensor([x1, y1, x2, y2]).unsqueeze(dim=0)
                roi_feature = roi_align(feature_map, [bbox])
                assert roi_feature.shape == (1, resnext_dim, pool_size, pool_size), \
                    'expected roi_feature.shape is  {} but got {}'.\
                    format((1, resnext_dim, pool_size, pool_size), roi_feature.shape)

                objects_mask[i, j] = 1
                temp_store.append(roi_feature)

        temp_store = torch.cat(temp_store, dim=0)
        assert temp_store.shape == (num_tracks, resnext_dim, pool_size, pool_size), \
            'expected temp_store.shape == {}, but got {}'.\
                format((num_tracks, resnext_dim, pool_size, pool_size), temp_store.shape)
        result_feature.append(temp_store)

    result_feature = torch.cat([item.unsqueeze(dim=0) for item in result_feature], dim=0)
    assert result_feature.shape == (time_span, num_tracks, resnext_dim, pool_size, pool_size), \
        'expected result_feature.shape == {}, but got {}'.\
            format((time_span, num_tracks, resnext_dim, pool_size, pool_size), result_feature.shape)

    np.save(visual_feature_path, result_feature)
    np.save(class_feature_path, class_store)
    np.save(objects_mask_path, objects_mask)
    with open(hastracks_path, 'wb') as f:
        pickle.dump(has_tracks, f)

    print('{video_name} has {x} objects'.format(video_name=video_name, x=len(tracks) if has_tracks else 0))


if __name__ == '__main__':
    # videos_dir = '/home/hanhuaye/PythonProject/train-video'
    videos_dir = '/home/hanhuaye/PythonProject/YouTubeClips'
    # data_dir = '/home/hanhuaye/PythonProject/gat_captioning/data/MSRVTT'
    data_dir = '/home/hanhuaye/PythonProject/gat_captioning/data/MSVD'
    # video_list = glob.glob(os.path.join(videos_dir, '*.mp4'))
    video_list = glob.glob(os.path.join(videos_dir, '*.avi'))
    save_dir = '/home/hanhuaye/PythonProject/opensource/deep_sort_pytorch/newoutput'
    has_process = glob.glob(os.path.join(os.path.join(data_dir, 'objects'), '*.npy'))
    has_process = [item.split('/')[-1].split('.')[0] for item in has_process]
    print(has_process)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnext101 = get_resnext101_32x8d(pretrained=True)
    resnext101 = resnext101.to(device)

    for video_path in video_list:
        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.')[0]
        if video_name in has_process: continue

        save_path = os.path.join(save_dir, video_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'info.log')
        cmd = obtain_tracks_cmd.format(video_path=video_path, save_path=save_path)
        cmd = cmd.split()
        if not os.path.exists(os.path.join(save_path, 'tracks.pkl')) and \
            not os.path.exists(os.path.join(save_path, 'length.pkl')):
            with open(log_path, 'w') as log:
                subprocess.call(cmd, stdout=log, stderr=log)

        get_visual_data(save_dir=save_path, data_dir=data_dir,
                        video_name=video_name, resnext101=resnext101,
                        device=device)


    print('========================Extraction finished!=============================')
        # import IPython
        # IPython.embed()