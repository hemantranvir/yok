"""Model file for Yolo v2."""

from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Reshape, Input, Lambda, Activation
from keras.layers import LeakyReLU
from keras.layers.merge import concatenate
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np


class ModelYolo():
    def __init__(self, input_shape, grid, box_no, batch_size, true_box_buffer,
                 class_names, anchors, predict=True):
        self.input_shape = input_shape
        self.grid_h, self.grid_w = grid
        self.box_no = box_no
        self.batch_size = batch_size
        self.true_box_buffer = true_box_buffer
        self.class_names = class_names
        self.class_weights    = np.ones(len(class_names), dtype='float32')
        self.anchors = anchors
        self.predict = predict
        self.true_boxes = Input(shape=(1, 1, 1, self.true_box_buffer, 4))

    def space_to_depth_x2(self, x):
        return tf.space_to_depth(x, block_size=2)

    def get_yolov2(self):
        input_image = Input(self.input_shape)
        # Layer 1
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(self.space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        out = Conv2D(self.box_no * (4 + 1 + len(self.class_names)), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
        out = Reshape((self.grid_h, self.grid_w, self.box_no, 4 + 1 + len(self.class_names)))(out)

        if not self.predict:
            out = Lambda(lambda args: args[0])([out, self.true_boxes])
            self.model = Model([input_image, self.true_boxes], out)
        else:
            self.model = Model(input_image, out)

        self.model.summary()
        return self.model

    def get_tinyyolov2(self):
        input_image = Input(self.input_shape)
        
        # Layer 1
        x = Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False)(input_image)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 2 - 5
        for i in range(0,4):
            x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False)(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Layer 6
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)
        
        # Layer 7
        x = Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 8
        x = Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # Layer 9
        out = Conv2D(self.box_no * (4 + 1 + len(self.class_names)), (1, 1), strides=(1, 1), padding='same', name='conv9')(x)
        out = Activation('linear')(out)
        out = Reshape((self.grid_h, self.grid_w, 5, 4 + 1 + len(self.class_names)))(out)

        if not self.predict:
            out = Lambda(lambda args: args[0])([out, self.true_boxes])
            self.model = Model([input_image, self.true_boxes], out)
        else:
            self.model = Model(input_image, out)

        self.model.summary()
        return self.model

    def custom_loss(self, y_true, y_pred):
        no_object_scale  = 1.0
        object_scale     = 5.0
        coord_scale      = 1.0
        class_scale      = 1.0
        warm_up_batches  = 0
    
        mask_shape = tf.shape(y_true)[:4]
    
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
    
        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, 5, 1])
    
        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
    
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
    
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.box_no, 2])
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell
        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half
        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half
    
        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
    
        true_box_conf = iou_scores * y_true[..., 4]
    
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
    
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * coord_scale
    
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
    
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half
    
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
    
        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half
    
        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)
    
        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * no_object_scale
    
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * object_scale
    
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_weights, true_box_class) * class_scale
    
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)
    
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, warm_up_batches),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * np.reshape(
                                                                    self.anchors, [1, 1, 1, self.box_no, 2]) * no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])
    
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
        loss = loss_xy + loss_wh + loss_conf + loss_class
    
        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
    
        """
        Debugging code
        """
        current_recall = nb_pred_box / (nb_true_box + 1e-6)
        total_recall = tf.assign_add(total_recall, current_recall)
    
        loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
        loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
        loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
        loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
        loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
        loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
        loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
        loss = tf.Print(loss, [total_recall / seen], message='Average Recall \t', summarize=1000)
    
        return loss
