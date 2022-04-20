import coco_metric
import numpy as np
from PIL import Image, ImageDraw


def detect(np_image_string,width,height,session):
    '''function to generate segmentation maks using Mask RCNN
    built using demo : https://colab.research.google.com/github/tensorflow/tpu/blob/master/models/official/mask_rcnn/mask_rcnn_demo.ipynb
    Returns an array of segmentation masks
    '''
    num_detections, detection_boxes, detection_classes, detection_scores, detection_masks, image_info = session.run(
    ['NumDetections:0', 'DetectionBoxes:0', 'DetectionClasses:0', 'DetectionScores:0', 'DetectionMasks:0', 'ImageInfo:0'],
    feed_dict={'Placeholder:0': np_image_string})

    num_detections = np.squeeze(num_detections.astype(np.int32), axis=(0,))
    detection_boxes = np.squeeze(detection_boxes * image_info[0, 2], axis=(0,))[0:num_detections]
    detection_scores = np.squeeze(detection_scores, axis=(0,))[0:num_detections]
    detection_classes = np.squeeze(detection_classes.astype(np.int32), axis=(0,))[0:num_detections]
    instance_masks = np.squeeze(detection_masks, axis=(0,))[0:num_detections]
    ymin, xmin, ymax, xmax = np.split(detection_boxes, 4, axis=-1)
    processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
    segmentations = coco_metric.generate_segmentation_from_masks(instance_masks, processed_boxes, height, width)
    return segmentations

def outline(image,segmentations,full):
    '''Function to generate outline using original image and segmentation mask
    Customise this function to generate differnt outlines and using different segmentation masks
    Returns: outline image'''
    if(not full):
        if(len(segmentations)>2):
            seg = segmentations[0] + segmentations[1] + segmentations[2]
        elif (len(segmentations)>1):
            seg = segmentations[0] + segmentations[1]
        else:
            seg = segmentations[0]
        seg[np.where(seg>0) ]=1
        for l in range(3):
            image[:,:,l]= image[:,:,l]*seg
        edges_out = cv2.Canny(seg,1,1)
    image = cv2.blur(image, (3,3))
    edges = cv2.Canny(image,80,200)
    edges = edges | edges_out
    return edges

'''Functions to generate constellation images from outlines'''
def draw_circle_white(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=250) 
def draw_circle_black(draw,c,dist):
    r = dist
    shape = [(c[0]-r,c[1]-r),(c[0]+r,c[1]+r)]
    draw.ellipse(shape,fill=0) 

'''Function to generate a dotted image from an outline'''    
def generate_image_dotted(edges,dist=40,dot= 2):
    '''dist (d) gives the distance between dots, dot is the radius of each dot (r)'''
    im = Image.fromarray(edges)
    # create rectangle image 
    draw = ImageDraw.Draw(im)   
    img_shape = edges.shape
    px = im.load()
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_black(draw,(j,i),dist)
                px[j,i]=255
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(px[j,i]==255):
                draw_circle_white(draw,(j,i),dot)
    return im
def add_noise(im,prob = 0.0001,dot =2 ):
    '''Function to add noise with a particular value of probability, Prob (p)'''
    draw = ImageDraw.Draw(im)
    img_shape = np.array(im).shape
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if(np.random.random()<prob):
                draw_circle_white(draw,(j,i),dot)
    return np.array(im)


