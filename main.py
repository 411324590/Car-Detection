# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from utils.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yolo.models.keras_yolo import yolo_head, yolo_boxes_to_corners
import cv2

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    """
    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=5, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape=(360., 640.), max_boxes=5, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]

    boxes = yolo_boxes_to_corners(box_xy, box_wh) #将X,Y,W,H转变为X1,X2,Y1,Y2

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    boxes = scale_boxes(boxes, image_shape) #缩放所预测的框，以便在图像上绘制。

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes

def predict(sess, image_file):
    """
    Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

    Arguments:
    sess -- tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image  .

    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    """

    image, image_data = preprocess_image(image_file, model_image_size=(608, 608))

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    colors = generate_colors(class_names)

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    #image.save(os.path.join("out", image_file), quality=90)
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    #imshow(image)
    return out_scores, out_boxes, out_classes,image

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (360., 640.)
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))#将最后的层特性转换为边界框参数。
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
VIDEO_NAME = "video_data/01.mp4"
video = cv2.VideoCapture(VIDEO_NAME)
if video.isOpened() == False:  # 异常检测
    print("Open Video Error!")
    exit()  # 退出
framecount = video.get(cv2.CAP_PROP_FRAME_COUNT)
for i in range(int(framecount)):
    success, frame = video.read()
    if i%4 == 0:
        #cv2.imshow("frame",frame)
        res = cv2.resize(frame, (640, 360))
        out_scores, out_boxes, out_classes,res = predict(sess, res)
        cv2.imshow("res",res)
        #cv2.imwrite(outpath+str((i+5)//5).zfill(4)+".jpg",res)
        if cv2.waitKey(1) & 0xff == ord('q'):
           exit()

cv2.destroyAllWindows()
# plt.ion()
# for i in range(1,140):
#     plt.cla()
#     num = str("%04d" % i)
#     out_scores, out_boxes, out_classes = predict(sess, num+'.jpg')
#     plt.pause(0.001)
# plt.ioff()
# plt.show()



