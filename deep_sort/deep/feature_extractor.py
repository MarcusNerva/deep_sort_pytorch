import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import logging
from PIL import Image
import torchvision.models as models
from .model import Net
from .pytorch_model import trans, get_resnext101_32x8d

class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        logger = logging.getLogger("root.tracker")
        logger.info("Loading weights from {}... Done!".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        


    def _preprocess(self, im_crops):
        """
        TODO:
            1. to float with scale from 0 to 1
            2. resize to (64, 128) as Market1501 dataset did
            3. concatenate to a numpy array
            3. to torch Tensor
            4. normalize
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch


    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()

class MyExtractor(object):
    def __init__(self):
        super(MyExtractor, self).__init__()
        self.net = get_resnext101_32x8d(pretrained=True)
        self.classifier = models.resnext101_32x8d(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(self.device)
        self.classifier.to(self.device)
        self.trans = trans

    # def _preprocess(self, im_crops):
    #     im_batch = torch.cat([self.trans(Image.fromarray(im)) for im in im_crops], dim=0).float()
    #     return im_batch

    def __call__(self, im_crops):
        # im_batch = self._preprocess(im_crops)
        features = []
        classes = []
        with torch.no_grad():
            for im in im_crops:
                im = Image.fromarray(im)
                im = self.trans(im).unsqueeze(0).to(self.device)
                feat = self.net(im)
                class_vec = self.classifier(im)
                feat = feat.squeeze().cpu().numpy()
                class_vec = class_vec.squeeze().cpu().numpy()
                features.append(feat)
                classes.append(class_vec)
            features = np.concatenate([item.unsqueeze(0) for item in features], axis=0)
            classes = np.concatenate([item.unsqueeze(0) for item in classes], axis=0)
        assert classes.shape == (len(im_crops), 1000), \
            'expect classes.shape == {}, but got {}'.\
                format((len(im_crops), 1000), classes.shape)
        return features, classes

if __name__ == '__main__':
    img = cv2.imread("demo.jpg")[:,:,(2,1,0)]
    extr = Extractor("checkpoint/ckpt.t7")
    feature = extr(img)
    print(feature.shape)

