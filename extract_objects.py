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

    with open(tracks_path, 'rb') as f:
        tracks = pickle.load(f)
    with open(length_path, 'rb') as f:
        frames_numb = pickle.load(f)

    frame_ids = np.linspace(start=2, stop=frames_numb, num=30, dtype=int)
    interval = frames_numb // 30

    visual_features = []
    for track in tracks:
        siz = track.get_length()
        if siz < interval: continue

        temp = []
        frames, features, positions, class_id = track['frame_ids'], track['features'], track['positions'], track['class_id']
        for id in frame_ids:
            if id not in frames:
                feat = np.ones(dim) * -1.0
            else:
                idx = frames.index(id)
                pos = np.array(positions[idx])
                visual = np.array(features[idx])
                class_vec = np.zeros(80)
                class_vec[class_id] = 1.0
                feat = np.concatenate([pos, visual, class_vec], axis=0)
                assert feat.size[0] == dim
            temp.append(feat)
        temp = np.stack(temp, axis=0)
        assert temp.size[0] == 30
        visual_features.append(temp)
    visual_features = np.stack(visual_features, axis=0)
    np.save(visual_feature_path, visual_features)



if __name__ == '__main__':
    videos_dir = '/home/hanhuaye/PythonProject/train-video'
    save_dir = '/home/hanhuaye/PythonProject/opensource/deep_sort_pytorch/output'
    data_dir = '/home/hanhuaye/PythonProject/rs_captioning/data'
    video_list = glob.glob(os.path.join(videos_dir, '*.mp4'))

    for video_path in video_list:
        video_name = video_list.split('/')[-1]
        video_name = video_name.split('.')[0]
        save_path = os.path.join(save_dir, video_name)
        log_path = os.path.join(save_path, 'info.log')
        obtain_tracks_cmd = obtain_tracks_cmd.format(video_path=video_path, save_path=save_path)
        obtain_tracks_cmd = obtain_tracks_cmd.split()
        with open(log_path, 'w') as log:
            subprocess.call(obtain_tracks_cmd, stdout=log, stderr=log)

        get_visual_data(save_dir=save_path, data_dir=data_dir, video_name=video_name)