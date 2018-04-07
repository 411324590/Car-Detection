# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from utils.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes
from yolo.models.keras_yolo import yolo_head
import cv2

def yolo_boxes_to_corners(box_xy, box_wh):
    #将X,Y,W,H转变为X_min,,Y_min,X_max,Y_max
    box_mins = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)

    return K.concatenate([
        box_mins[..., 1:2],  # y_min
        box_mins[..., 0:1],  # x_min
        box_maxes[..., 1:2],  # y_max
        box_maxes[..., 0:1]  # x_max
    ])

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=0.6):

    box_scores = box_confidence * box_class_probs#(19, 19, 5, 80)

    box_classes = K.argmax(box_scores, axis=-1)#维度减一
    box_class_scores = K.max(box_scores, axis=-1)#维度减一
    filtering_mask = box_class_scores >= threshold

    scores = tf.boolean_mask(box_class_scores, filtering_mask)#维度减（1维（None,1））
    boxes = tf.boolean_mask(boxes, filtering_mask)#维度减（2维（None，4）
    classes = tf.boolean_mask(box_classes, filtering_mask)#维度减（1维（None,1））

    return scores, boxes, classes

def scale_boxes(boxes, image_shape):
    #根据输入图尺寸标准化所预测的框的坐标值（boxes的X_min,,Y_min,X_max,Y_max），以便在图像上绘制
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])#K.stack()将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量
    image_dims = K.reshape(image_dims, [1, 4])#K.reshape()将张量的shape变换为指定shape
    boxes = boxes * image_dims
    return boxes

def yolo_non_max_suppression(scores, boxes, classes, max_boxes=5, iou_threshold=0.5):


    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # 应用于tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # 初始化 max_boxes_tensor

    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    # xi1 = max(box1[0],box2[0]) ，yi1 = max(box1[1],box2[1])，xi2 = min(box1[2],box2[2])，yi2 = min(box1[3],box2[3])，
    # inter_area = (yi2-yi1)*(xi2-xi1)
    # box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    # box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    # union_area = box1_area + box2_area - inter_area
    # iou = inter_area / union_area

    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape=(360., 640.), max_boxes=5, score_threshold=0.6, iou_threshold=0.5):
    """
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    #该函数将接收深度CNN (19x19x5x85维度编码)的输出，并使用刚刚实现的函数对所有的boxes进行过滤。
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]#获得值

    boxes = yolo_boxes_to_corners(box_xy, box_wh) #将X,Y,W,H转变为X1,X2,Y1,Y2

    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)#调用yolo_filter_boxes

    boxes = scale_boxes(boxes, image_shape) #根据输入图尺寸标准化所预测的框的坐标值（boxes的X1,X2,Y1,Y2），以便在图像上绘制。

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)#调用yolo_non_max_suppression

    return scores, boxes, classes

def predict(sess, image):
    """
    sess -- tensorflow/Keras session containing the YOLO graph
    image -- name of an image  .

    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    """

    image, image_data = preprocess_image(image, model_image_size=(608, 608))#将图片resize并二值化

    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    #模型使用BatchNorm需要在feed_dict {K.learning_phase(): 0}中传递一个额外的占位符。

    colors = generate_colors(class_names)#返回一个和class_names一样长的颜色列表

    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)#在image上画框画标签
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)#将image转为CV2的对象
    #image.save(os.path.join("out", image_file), quality=90)
    #output_image = scipy.misc.imread(os.path.join("out", image_file))
    #imshow(image)
    return out_scores, out_boxes, out_classes,image

sess = K.get_session()
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")#读取anchors_box的宽高
image_shape = (360., 640.)
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))#将DarkNet最后的输出转换为我们想要的输出参数格式。
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)#送入yolo_eval，得出我们在图片上画框前所需要的数据
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
        #res =cv2.resize(res, (1280, 720))
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



