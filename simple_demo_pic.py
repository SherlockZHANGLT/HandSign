#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import time
import copy

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
    frame=cv.imread('D:/test/monkey/monkey_IMG_22c21d866-4d5d-11ea-b58b-0242ac1c0002.jpg')
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
    while True:
        for i in range(num_detections):
            # 検出結果可視化 ###################################################
            x1, y1 = int(bbox[1] * frame_width), int(bbox[0] * frame_height)
            x2, y2 = int(bbox[3] * frame_width), int(bbox[2] * frame_height)
            cv.putText(
                debug_image, 'ID:' + str(class_id) + ' ' +
                labels[class_id][0] + ' ' + '{:.3f}'.format(score),
                (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                cv.LINE_AA)
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(0)
        if key == 27:  # ESC
            break
        # 画面反映 #############################################################
        cv.imshow('NARUTO HandSignDetection Simple Demo', debug_image)
if __name__ == '__main__':
    main()
