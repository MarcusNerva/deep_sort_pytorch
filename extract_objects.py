import pickle
import subprocess
import numpy as np
import glob
import os


obtain_tracks_cmd = 'python /home/hanhuaye/PythonProject/opensource/deep_sort_pytorch/yolov3_deepsort.py ' \
                    '{video_path} --save_path {save_path}'
dim = 4 + 2048 + 80

def get_visual_data(save_dir, data_dir, video_name):
    tracks_path = os.path.join(save_dir, 'tracks.pkl')
    length_path = os.path.join(save_dir, 'length.pkl')
    visual_feature_dir = os.path.join(data_dir, 'objects')
    visual_feature_path = os.path.join(visual_feature_dir, video_name + '.npy')
    hastracks_path = os.path.join(visual_feature_dir, video_name + '_hastracks.pkl')

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    with open(length_path, 'rb') as f:
        frames_numb = pickle.load(f)

    # frame_ids = np.linspace(start=2, stop=frames_numb - 1, num=30, dtype=int)
    interval = frames_numb // 30

    has_tracks = True
    visual_features = []
    fake_feature = np.concatenate([np.ones([1, 30, dim - 80]) * -1.0, np.zeros([1, 30, 80])], axis=-1)
    for track in tracks:
        siz = len(track['frame_ids'])
        if siz < 3 * interval: continue

        temp = []
        frames, features, positions, class_id = track['frame_ids'], track['features'], track['positions'], track['class_id']
        frame_ids = np.linspace(start=0, stop=len(frames) - 1, num=30, dtype=int)
        for id in frame_ids:
            pos = np.array(positions[id])
            visual = np.array(features[id])
            class_vec = np.zeros(80)
            class_vec[class_id] = 1.0
            feat = np.concatenate([pos, visual, class_vec], axis=0)
            assert feat.shape[0] == dim
            temp.append(feat)
        temp = np.concatenate([feat[np.newaxis, ...] for feat in temp], axis=0)
        assert temp.shape[0] == 30 and temp.shape[1] == dim and len(temp.shape) == 2
        visual_features.append(temp)
    if len(visual_features) > 0:
        visual_features = np.concatenate([vf[np.newaxis, ...] for vf in visual_features], axis=0)
    else:
        has_tracks = False
        visual_features = fake_feature

    assert len(visual_features.shape) == 3 and visual_features.shape[1] == 30 and visual_features.shape[-1] == dim
    np.save(visual_feature_path, visual_features)
    with open(hastracks_path, 'wb') as f:
        pickle.dump(has_tracks, f)
    print('{video_name} has {x} objects'.format(video_name=video_name, x=visual_features.shape[0] if has_tracks else 0))



if __name__ == '__main__':
    # videos_dir = '/home/hanhuaye/PythonProject/train-video'
    videos_dir = '/home/hanhuaye/PythonProject/YouTubeClips'
    # data_dir = '/home/hanhuaye/PythonProject/rs_captioning/data/MSRVTT'
    data_dir = '/home/hanhuaye/PythonProject/rs_captioning/data/MSVD'
    # video_list = glob.glob(os.path.join(videos_dir, '*.mp4'))
    video_list = glob.glob(os.path.join(videos_dir, '*.avi'))
    save_dir = '/home/hanhuaye/PythonProject/opensource/deep_sort_pytorch/output'
    has_process = glob.glob(os.path.join(os.path.join(data_dir, 'objects'), '*.npy'))
    has_process = [item.split('/')[-1].split('.')[0] for item in has_process]
    print(has_process)

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
            os.path.exists(os.path.join(save_path, 'length.pkl')):
            with open(log_path, 'w') as log:
                subprocess.call(cmd, stdout=log, stderr=log)

        get_visual_data(save_dir=save_path, data_dir=data_dir, video_name=video_name)


    print('========================Extraction finished!=============================')
        # import IPython
        # IPython.embed()