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
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])
resnext_dim = 2048


def process_pos(pos, width, height, new_width, new_height):
    pos[0] = max(0, pos[0])
    pos[1] = max(0, pos[1])
    pos[2:] = pos[:2] + pos[2:]

    x1, y1, x2, y2 = pos[0], pos[1], pos[2], pos[3]
    x1 = min(x1, width)
    x2 = min(x2, width)
    y1 = min(y1, height)
    y2 = min(y2, height)

    x_scale, y_scale = float(new_width) / width, float(new_height) / height
    x1, y1, x2, y2 = x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale

    return np.array([x1, y1, x2, y2], dtype=float)


def get_visual_data(save_dir, data_dir, video_dir, video_name, resnext101, device):
    tracks_path = os.path.join(save_dir, 'tracks.pkl')
    length_path = os.path.join(save_dir, 'length.pkl')
    video_path = os.path.join(video_dir, video_name + '.avi')
    visual_feature_dir = os.path.join(data_dir, 'objects')
    class_feature_dir = os.path.join(data_dir, 'category')
    visual_feature_path = os.path.join(visual_feature_dir, video_name + '.npy')
    class_feature_path = os.path.join(class_feature_dir, video_name + '.npy')
    objects_mask_path = os.path.join(visual_feature_dir, video_name + '_mask.pkl')
    hastracks_path = os.path.join(visual_feature_dir, video_name + '_hastracks.pkl')

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    with open(length_path, 'rb') as f:
        frames_numb = pickle.load(f)

    interval = frames_numb // 15
    tracks = [item for item in tracks if len(item['frame_ids']) >= interval]
    tracks = sorted(tracks, key=lambda x: len(x['frame_ids']), reverse=True)
    if len(tracks) > 20:
        tracks = tracks[:20]
    num_tracks = len(tracks)

    if num_tracks == 0:
        has_tracks = False
        visual_features = np.zeros([15, 1, 2048, 4, 4])
        objects_mask = np.zeros([15, 1])
        class_features = np.zeros([1, 1000], dtype=int)

        np.save(visual_feature_path, visual_features)
        np.save(class_feature_path, class_features)
        with open(objects_mask_path, 'wb') as f:
            pickle.dump(objects_mask, f)
        with open(hastracks_path, 'wb') as f:
            pickle.dump(has_tracks, f)
        return

    frames_store = []
    import cv2
    vdo = cv2.VideoCapture()
    vdo.open(video_path)
    im_width, im_height = int(vdo.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while vdo.grab():
        _, ori_im = vdo.retrieve()
        im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        frames_store.append(trans(im).unsqueeze(0))
    new_height, new_width = frames_store[0].shape[-2], frames_store[0].shape[-1]
    assert len(frames_store) == frames_numb, \
        'expect len(frames_store) == frames_numb, ' \
        'but got {} and {}'.format(len(frames_store), frames_numb)

    visit = np.zeros(frames_numb + 5, dtype=int)
    for track in tracks:
        frames_ids = track['frame_ids']
        for id in frames_ids:
            new_id = id - 1
            assert new_id >= 0
            visit[new_id] = 1

    class_store = []
    for track in tracks:
        class_id = track['class_id']
        class_vec = np.zeros(1000, dtype=int)
        class_vec[class_id] = 1
        class_store.append(class_vec)
    class_store = np.concatenate([item[np.newaxis, ...] for item in class_store], axis=0)
    assert class_store.shape == (num_tracks, 1000), \
        'expected class_store.shape == ({}, 1000), but got {}'.\
            format(num_tracks, class_store.shape)

    time_span = 15
    step = max(int(visit.sum()) // time_span, 1)
    anchor_idxs = []
    has_tracks = True
    cnt = 0
    for i in range(frames_numb):
        if visit[i] == 0: continue
        cnt += 1
        if cnt % step == 0:
            anchor_idxs.append(i)
    origin_anchors_len = len(anchor_idxs)
    idxs = np.linspace(0, len(anchor_idxs), time_span, endpoint=False, dtype=int).tolist()
    anchor_idxs = [anchor_idxs[i] for i in idxs] if origin_anchors_len > time_span else anchor_idxs
    if origin_anchors_len >= time_span:
        assert len(anchor_idxs) == time_span, \
            'expected len(anchor_idxs) == time_span, but got {}, and now step is {}'.\
                format(len(anchor_idxs), step)
    else:
        assert int(visit.sum()) < time_span, \
            'expect visit.sum() < time_span, but get {} and step is {}'.format(int(visit.sum()), step)

    pool_size = 4
    spatial_scale = 1.0 / 32.0
    fake_feature = np.zeros([1, resnext_dim, pool_size, pool_size])
    roi_align = RoIAlign((pool_size, pool_size), spatial_scale=spatial_scale, sampling_ratio=1).to(device)

    result_feature = []
    objects_mask = np.zeros([time_span, len(tracks)])
    for i in range(time_span):
        idx = anchor_idxs[i] if i < len(anchor_idxs) else None
        temp_store = []
        feature_map = resnext101(frames_store[idx].to(device)) if idx is not None else None  # (1, 2048, H/32, W/32)
        for j, item in enumerate(tracks):
            frames_ids, positions, class_id = item['frame_ids'], item['positions'], item['class_id']
            if idx is None or idx not in frames_ids:
                objects_mask[i, j] = 0
                temp_store.append(fake_feature)
            else:
                ptr = frames_ids.index(idx)
                position = positions[ptr]
                position = process_pos(position, im_width, im_height, new_width, new_height)
                x1, y1, x2, y2 = position[0], position[1], position[2], position[3]
                bbox = torch.FloatTensor([x1, y1, x2, y2]).unsqueeze(dim=0).to(device)
                roi_feature = roi_align(feature_map, [bbox])
                assert roi_feature.shape == (1, resnext_dim, pool_size, pool_size), \
                    'expected roi_feature.shape is  {} but got {}'.\
                    format((1, resnext_dim, pool_size, pool_size), roi_feature.shape)

                objects_mask[i, j] = 1
                temp_store.append(roi_feature.detach().cpu().numpy())

        temp_store = np.concatenate([item for item in temp_store], axis=0)
        assert temp_store.shape == (num_tracks, resnext_dim, pool_size, pool_size), \
            'expected temp_store.shape == {}, but got {}'.\
                format((num_tracks, resnext_dim, pool_size, pool_size), temp_store.shape)
        result_feature.append(temp_store)

    result_feature = np.concatenate([item[np.newaxis, ...] for item in result_feature], axis=0)
    assert result_feature.shape == (time_span, num_tracks, resnext_dim, pool_size, pool_size), \
        'expected result_feature.shape == {}, but got {}'.\
            format((time_span, num_tracks, resnext_dim, pool_size, pool_size), result_feature.shape)
    assert objects_mask[len(anchor_idxs):, ...].sum() == 0., \
        'expect 0. in objects_mask[len(anchor_idxs):, ...], but got {}'.\
            format(objects_mask[len(anchor_idxs):, ...].sum())

    np.save(visual_feature_path, result_feature)
    np.save(class_feature_path, class_store)
    with open(objects_mask_path, 'wb') as f:
        pickle.dump(objects_mask, f)
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

        get_visual_data(save_dir=save_path, video_dir=videos_dir,
                        data_dir=data_dir,
                        video_name=video_name, resnext101=resnext101,
                        device=device)


    print('========================Extraction finished!=============================')
        # import IPython
        # IPython.embed()