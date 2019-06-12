import argparse
import json
import numpy as np
from yok.utils import *
from matplotlib.pyplot import imread, imshow
import matplotlib.pyplot as plt

from keras import backend as K
K.set_learning_phase(1)

import cv2
from tqdm import tqdm

from yok.models import ModelYolo

def get_argparser():
    argparser = argparse.ArgumentParser(description="Load Yolov2/TinyYolov2 model and run detection",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-c', '--config_path', help='path to json config file',
                           default=os.path.join('./config_predict.json'))
    argparser.add_argument('-i', '--input', help='path to input image/video file',
                           default=os.path.join('videos', 'conference.mp4'))
    argparser.add_argument('-o', '--output', help='path to output image/video file',
                           default=os.path.join('videos', 'conference_detect.avi'))
    argparser.add_argument('-l', '--log', help='path to output csv file',
                           default=os.path.join('logs', 'number_count.csv'))
    argparser.print_help()
    return argparser

def main(args):
    with open(args.config_path) as f:
        data = json.loads(f.read())

    anchors = np.array(data['COCO']['anchors'])
    class_names = data['COCO']['class_names']
    model_name = data['COCO']['model']
    image_h = data['COCO']['image_h']
    image_w = data['COCO']['image_w']
    grid_h = data['COCO']['grid_h']
    grid_w = data['COCO']['grid_w']
    box_no = data['COCO']['box_no']
    true_box_buffer = data['COCO']['true_box_buffer']
    batch_size = data['COCO']['batch_size']
    weights_file = data['COCO']['weights']
    target_class = data['COCO']['target_class']

    yolo = ModelYolo((image_h, image_w, 3), (grid_h, grid_w), box_no,
                     batch_size, true_box_buffer, class_names, anchors)

    if model_name == 'Yolov2':
        model = yolo.get_yolov2()
    elif model_name == 'TinyYolov2':
        model = yolo.get_tinyyolov2()
    else:
        raise ValueError('Specified model is not supported,'
                          'please specify Yolov2 or TinyYolov2')

    model.load_weights(weights_file)

    #video_reader = cv2.VideoCapture(0)

    input_file = args.input
    output_file = args.output
    log_file = args.log

    video_reader = cv2.VideoCapture(input_file)
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    if nb_frames != 1:
        video_writer = cv2.VideoWriter(output_file,
                                       cv2.VideoWriter_fourcc(*'mp4v'),
                                       fps,
                                       (frame_w, frame_h))
        f = open(log_file, mode='w+')

    sess = K.get_session()

    files = list()
    file_name = '/tmp/temp.jpg'
    files.append(file_name)

    for i in tqdm(range(nb_frames), desc='Progress'):
        ret, image = video_reader.read()

        cv2.imwrite(file_name, image)

        image_test, image_data = preprocess_test_image(file_name, model_image_size = (416, 416))

        image_shape = image_test.size
        image_shape = tuple(float(i) for i in reversed(image_test.size))

        yolo_outputs = yolo_head(model.output, anchors, len(class_names))
        scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
            model.input: image_data
        })

        print('Found {} boxes for {}'.format(len(out_boxes), input_file))

        colors = get_spaced_colors(len(class_names))
        count = draw_boxes(image_test, out_scores, out_boxes, out_classes, class_names, colors, target_class)

        if nb_frames != 1:
            file_name1 = '/tmp/temp1.jpg'
            files.append(file_name1)
            image_test.save(file_name1, quality=90)
            output_image = imread(file_name1)
            im2 = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

            video_writer.write(im2)
            if i%fps == 0:
                info = '{}, {}\n'.format(int(i/fps), count)
                f.write(info)
                f.flush()
        else:
            image_test.save(os.path.join(output_file), quality=90)

    video_reader.release()
    if nb_frames != 1:
        video_writer.release()
        f.close()

    for f_ in files:
        if os.path.exists(f_):
            os.remove(f_)

if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    main(args)
