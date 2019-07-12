"""

        Non maxima suppression on WSI results
        
        Uses a kdTree to improve speed. This will only work reasonably for same-sized objects.
        
        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019

"""


import pickle
import numpy as np
import sys
from sklearn.neighbors import KDTree


def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
    if (det_thres is not None): # perform thresholding
        to_keep = scores>det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        
    
    if boxes.shape[-1]==6: # BBOXES
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
    else:
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]

        
    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        id = sorted_ids[0]
        ids_to_keep.append(id)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[id]).nonzero()[0])

    return boxes[ids_to_keep]

def nms(result_boxes, det_thres=None):
    
    for filekey in result_boxes.keys():
        arr = np.array(result_boxes[filekey])
        if arr is not None and isinstance(arr, np.ndarray) and (arr.shape[0] == 0):
            continue
        if (det_thres is not None):
            before = np.sum(arr[:,-1]>det_thres)
        if (arr.shape[0]>0):
            try:
                arr = non_max_suppression_by_distance(arr, arr[:,-1], 25, det_thres)
            except:
                pass

        result_boxes[filekey] = arr
    
    return result_boxes