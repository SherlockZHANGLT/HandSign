#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import csv
import time
import copy
import datetime
import cv2 as cv
import numpy as np
import tensorflow as tf

def run_inference_single_image(image, inference_func):
    tensor = tf.convert_to_tensor(image)
    output = inference_func(tensor)

    output['num_detections'] = int(output['num_detections'][0])
    output['detection_classes'] = output['detection_classes'][0].numpy()
    output['detection_boxes'] = output['detection_boxes'][0].numpy()
    output['detection_scores'] = output['detection_scores'][0].numpy()
    return output

def main():
    width=960
    height=540
    model_path='model/EfficientDetD0/saved_model'
    score_th=0.75

    DEFAULT_FUNCTION_KEY = 'serving_default'
    loaded_model = tf.saved_model.load(model_path)
    inference_func = loaded_model.signatures[DEFAULT_FUNCTION_KEY]

    with open('setting/labels.csv', encoding='utf8') as f:
        labels = csv.reader(f)
        labels = [row for row in labels]

    file_pathname='D:/test'
    for filename in os.listdir(file_pathname):
        print(filename)
        starttime=time.time()
        frame = cv2.imread(file_pathname+'/'+filename)
        frame_width,frame_height=frame.shape[1],frame.shape[0]
        debug_image=copy.deepcopy(frame)
        frame = frame[:, :, [2, 1, 0]]  # BGR2RGB
        image_np_expanded = np.expand_dims(frame, axis=0)
        output = run_inference_single_image(image_np_expanded, inference_func)
        num_detections = output['num_detections']
        for i in range(num_detections):
            score = output['detection_scores'][i]
            bbox = output['detection_boxes'][i]
            class_id = output['detection_classes'][i].astype(np.int)
            if score < score_th:
                continue
        print(class_id)
        endtime=time.time()
        print(endtime-starttime)

if __name__ == '__main__':
    main()
