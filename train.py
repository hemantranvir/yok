import json
import argparse
from lib.models import ModelYolo
from lib.preprocessing import parse_annotation, BatchGenerator
from keras.optimizers import Adam
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def normalize(image):
    image = image / 255.
    return image

def get_argparser():
    argparser = argparse.ArgumentParser(description="Train Yolov2/TinyYolov2 model",
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('-c', '--config_path', help='path to json config file',
                           default=os.path.join('./config_train.json'))
    argparser.add_argument('-i', '--input', help='path to pre-trained h5 file',
                           default=os.path.join('h5_models/pre_trained', 'yolov2.h5'))
    argparser.add_argument('-o', '--output', help='path to output trained file',
                           default=os.path.join('h5_models/trained', 'yolov2_coco.h5'))
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
    batch_size = data['COCO']['batch_size']
    true_box_buffer = data['COCO']['true_box_buffer']
    train_image_folder = data['COCO']['train_image_folder']
    train_annot_folder = data['COCO']['train_annot_folder']
    valid_image_folder = data['COCO']['valid_image_folder']
    valid_annot_folder = data['COCO']['valid_annot_folder']
    epochs = data['COCO']['epochs']

    yolo = ModelYolo((image_h, image_w, 3), (grid_h, grid_w), box_no,
                     batch_size, true_box_buffer, class_names, anchors, predict=False)

    if model_name == 'Yolov2':
        model = yolo.get_yolov2()
    elif model_name == 'TinyYolov2':
        model = yolo.get_tinyyolov2()
    else:
        raise ValueError('Specified model is not supported,'
                          'please specify Yolov2 or TinyYolov2')
    
    model.load_weights(args.input)
    
    ##TODO: For faster execution the randomization of last layers are skipped for now

    output_file = args.output
    
    generator_config = {
        'image_h'         : image_h,
        'image_w'         : image_w,
        'grid_h'          : grid_h,
        'grid_w'          : grid_w,
        'box_no'          : box_no,
        'class_names'     : class_names,
        'class_no'        : len(class_names),
        'anchors'         : anchors,
        'batch_size'      : batch_size,
        'true_box_buffer' : true_box_buffer,
    }
    
    
    train_imgs, seen_train_labels = parse_annotation(train_annot_folder, train_image_folder, labels=class_names)
    train_batch = BatchGenerator(train_imgs, generator_config, norm=normalize)
    
    valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder, valid_image_folder, labels=class_names)
    valid_batch = BatchGenerator(valid_imgs, generator_config, norm=normalize, jitter=False)
    
    #early_stop = EarlyStopping(monitor='val_loss',
    #                           min_delta=0.001,
    #                           patience=3,
    #                           mode='min',
    #                           verbose=1)
    
    checkpoint = ModelCheckpoint(output_file,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True,
                                 mode='min',
                                 period=5)
    
    optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss=yolo.custom_loss, optimizer=optimizer)
    
    print('Length of train batch: ', len(train_batch))
    print('Length of val batch: ', len(valid_batch))
    
    model.fit_generator(generator        = train_batch,
                        steps_per_epoch  = len(train_batch),
                        epochs           = epochs,
                        verbose          = 1,
                        validation_data  = valid_batch,
                        validation_steps = len(valid_batch),
                        callbacks        = [checkpoint],
                        max_queue_size   = 3)
    
    print('Completed fitting model')

if __name__ == '__main__':
    argparser = get_argparser()
    args = argparser.parse_args()
    main(args)
