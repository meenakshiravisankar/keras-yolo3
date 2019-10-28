# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
import numpy as np
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import pytesseract

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (256, 256),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        vehicle_labels = []
        pedestrian_labels = []

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            # thresholding to avoid misclassification of same object
            if score < 0.5 :
                continue
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            label_vehicles = ["car", "motorbike", "bus", "truck"]
            if predicted_class in label_vehicles :
                vehicle_labels.append([predicted_class, left, top, right, bottom])
            elif predicted_class == "person" :
                pedestrian_labels.append([predicted_class, left, top, right, bottom])

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image, np.array(vehicle_labels), np.array(pedestrian_labels)

    def close_session(self):
        self.sess.close()

def get_videotime(frame) :
    """
    Crops the time-in-video information and returns the timestamp
    
    Parameters
    ----------
    frame - Entire traffic image with timestamp

    Returns
    -------
    text - Text with timestamp details
    """
    # Crop the time portion
    frame = frame[0:25,0:220]
    # perform ocr
    text = pytesseract.image_to_string(frame)  
    return text
        
def detect_video(yolo, video_path, output_path=""):
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    current_vehicles = 0
    total_left_vehicles = 0
    total_right_vehicles = 0
    
    current_pedestrians = 0
    total_pedestrians = 0
    
    activity_log = np.array([])
    save_log_time = 0
    activity_log_path = "../files/results/"

    if not os.path.exists(activity_log_path) :
        os.makedirs(activity_log_path)

    while True:
        return_value, frame = vid.read()
        if not return_value :
            break
        time_in_video = get_videotime(frame)
        image = Image.fromarray(frame)
        image, vehicle_labels, pedestrian_labels = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        
        # Display results
        height = result.shape[0]
        width = result.shape[1]

        
        current_vehicles = vehicle_labels.shape[0]
        current_pedestrians = pedestrian_labels.shape[0]
        
        detection_threshold = int(width/2)-200 
        vehicle_threshold = 15
        pedestrain_threshold = 5

        # count if the vehicle has just entered
        for vehicle in vehicle_labels :
            col = (int(vehicle[1])+int(vehicle[3]))/2
            row = (int(vehicle[2])+int(vehicle[4]))/2
            
            if abs(col-detection_threshold) < vehicle_threshold :
                if row>height/2 :
                    total_left_vehicles+=1
                else :
                    total_right_vehicles+=1
                vehicle = np.append(vehicle, time_in_video)   
                activity_log = np.append(activity_log, vehicle)     

        for pedestrian in pedestrian_labels :
            col = (int(pedestrian[1])+int(pedestrian[3]))/2
            row = (int(pedestrian[2])+int(pedestrian[4]))/2
            if abs(row-height/2) < pedestrain_threshold :
                total_pedestrians+=1
                pedestrian = np.append(pedestrian, time_in_video)
                activity_log = np.append(activity_log, pedestrian)


        size = 256
        font_size = 0.6
        img = np.zeros((size,size,3), np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Vehicles',(5,30), font, font_size,(255,255,255),2,8)
        cv2.putText(img,'Current Frame : '+str(current_vehicles),(0,60), font, font_size,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'Left Lane : '+str(total_left_vehicles),(5,90), font, font_size,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'Right Lane : '+str(total_right_vehicles),(5,120), font, font_size,(255,255,255),2,cv2.LINE_AA)
        img = cv2.line(img,(5,130),(100,130),(255,255,255),1)

        cv2.putText(img,'Pedestrians',(5,150), font, font_size,(255,255,255),2,8)
        cv2.putText(img,'Current Frame : '+str(current_pedestrians),(5,180), font, font_size,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'Total   : '+str(total_pedestrians),(5,210), font, font_size,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(img,'Total   : '+str(total_pedestrians),(5,240), font, font_size,(255,255,255),2,cv2.LINE_AA)
        

        cv2.imshow('Traffic Analysis',img)
        image = np.array(image)
        image[0:size,width-size:width] = img
        cv2.imshow('result', image)

        print(activity_log, save_log_time)
        if(save_log_time%10) :
            activity_log = np.array(activity_log)
            np.savetxt(os.path.join(activity_log_path,"log.txt"), activity_log, fmt="%s")

        save_log_time+=1
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()

