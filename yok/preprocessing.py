import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from yok.utils import BoundBox, normalize, bbox_iou


def parse_annotation(ann_dir, img_dir, labels=[]):
    count = 0
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        if count > 2:
            break

        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        print(obj['name'])
                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            count = count + 1

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images,
                 config,
                 shuffle=True,
                 jitter=True,
                 norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        self.anchors = [BoundBox(0, 0, config['anchors'][2 * i], config['anchors'][2 * i + 1]) for i in
                        range(int(len(config['anchors']) // 2))]

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.aug_pipe = iaa.Sequential(
            [
                sometimes(iaa.Affine(
                )),
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   iaa.MedianBlur(k=(3, 11)),
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               ]),
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images)) / self.config['batch_size']))

    def __getitem__(self, idx):
        l_bound = idx * self.config['batch_size']
        r_bound = (idx + 1) * self.config['batch_size']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['batch_size']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['image_h'], self.config['image_w'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config['true_box_buffer'],
                            4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['grid_h'], self.config['grid_w'], self.config['box_no'],
                            4 + 1 + self.config['class_no']))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)

            true_box_index = 0

            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['class_names']:
                    center_x = .5 * (obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['image_w']) / self.config['grid_w'])
                    center_y = .5 * (obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['image_h']) / self.config['grid_h'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['grid_w'] and grid_y < self.config['grid_h']:
                        obj_indx = self.config['class_names'].index(obj['name'])

                        center_w = (obj['xmax'] - obj['xmin']) / (
                        float(self.config['image_w']) / self.config['grid_w'])
                        center_h = (obj['ymax'] - obj['ymin']) / (
                        float(self.config['image_h']) / self.config['grid_h'])

                        box = [center_x, center_y, center_w, center_h]

                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0,
                                               0,
                                               center_w,
                                               center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)

                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                        b_batch[instance_count, 0, 0, 0, true_box_index] = box

                        true_box_index += 1
                        true_box_index = true_box_index % self.config['true_box_buffer']

            if self.norm != None:
                x_batch[instance_count] = self.norm(img)
            else:
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:, :, ::-1], (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']),
                                      (255, 0, 0), 3)
                        cv2.putText(img[:, :, ::-1], obj['name'],
                                    (obj['xmin'] + 2, obj['ymin'] + 12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0, 255, 0), 2)

                x_batch[instance_count] = img

            instance_count += 1

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)

    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: print
        'Cannot find ', image_name

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            max_offx = (scale - 1.) * w
            max_offy = (scale - 1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)

            image = image[offy: (offy + h), offx: (offx + w)]

            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)

            image = self.aug_pipe.augment_image(image)

        image = cv2.resize(image, (self.config['image_h'], self.config['image_w']))
        image = image[:, :, ::-1]

        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)

                obj[attr] = int(obj[attr] * float(self.config['image_w']) / w)
                obj[attr] = max(min(obj[attr], self.config['image_w']), 0)

            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)

                obj[attr] = int(obj[attr] * float(self.config['image_h']) / h)
                obj[attr] = max(min(obj[attr], self.config['image_h']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['image_w'] - obj['xmax']
                obj['xmax'] = self.config['image_w'] - xmin

        return image, all_objs
