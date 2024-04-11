from SORT import SORT, KalmanBoxTracker
import numpy as np
from torchvision.models import mobilenet_v3_small
import torch.nn as nn
from typing import List
from utils import IoU
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import cv2

def compare_boxes_with_embeds(detections, trackers, detections_embeds, trackers_embeds, iou_thresh=0.3):

    iou_matrix = np.zeros(shape=(len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = 0.8*IoU(det,trk) + 0.2*cosine_similarity([np.squeeze(detections_embeds[d])], [np.squeeze(trackers_embeds[t])])[0][0]
    
    # calculate maximum iou for each pair through hungarian algorithm
    row_id, col_id = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_id,col_id]))
    # geting matched ious
    iou_values = np.array([iou_matrix[row_id,col_id] for row_id,col_id in matched_indices])
    best_indices = matched_indices[iou_values > iou_thresh]

    unmatched_detection_indices = np.array([d for d in range(len(detections)) if d not in best_indices[:,0]])  
    unmatched_trackers_indices = np.array([t for t in range(len(trackers)) if t not in best_indices[:,1]])

    return best_indices,unmatched_detection_indices,unmatched_trackers_indices

# def transform_bbox(bbox):
#     x1, y1, x2, y2 = bbox
#     w = x2 - x1
#     h = y2 - y1
#     y1 += h
#     y2 += h
#     x1 += w/2
#     x2 += w/2
#     return [x1, y1, x2, y2]

def cut_preprocessed_bbox(frame, bbox, processing):
    image_w, image_h = frame.shape[1], frame.shape[0]
    x1, y1, x2, y2 = bbox
    cut = frame[max(int(y1), 0):min(int(y2), image_h), 
                max(int(x1), 0):min(int(x2), image_w)]
    return processing(cut)[None, :, :, :]

class StrongTracker(SORT):
    def __init__(self, threshold=0.3, max_age=4) -> None:
        super().__init__(threshold, max_age)
        self.extractor = nn.Sequential(*list(mobilenet_v3_small(pretrained=True).children())[:-1])
        self.processing = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def update(self, dets: List[List], frame) -> None:
        # get tracks prediction bboxes
        self.trackers = [tracker for tracker in self.trackers if not np.any(np.isnan(tracker.predict()))]

        trks = [tracker.get_state()[0].tolist() for tracker in self.trackers]
        frame = cv2.resize(frame, (1000, 800))
        # caclucalte embeddings
        trks_embeds = [self.extractor(cut_preprocessed_bbox(frame, trk, self.processing)).detach().numpy() for trk in trks]
        dets_embeds = [self.extractor(cut_preprocessed_bbox(frame, det, self.processing)).detach().numpy() for det in dets]

        # compare detection and tracks predictions based on IOU
        matched, unmatched_dets, unmatched_tracks = compare_boxes_with_embeds(dets, 
                                                                              trks, 
                                                                              dets_embeds, 
                                                                              trks_embeds, 
                                                                              iou_thresh=self.iou_threshold)

        out = {}
        # Then we will update the kalman filter with measurements
        # for each detection we maintain seperate filter
        for detection_num, tracker_num in matched:
            self.trackers[tracker_num].update(dets[detection_num])
            # out[detection_num] = tracker_num
            out[detection_num] = self.trackers[tracker_num].id
            
        # For all unmatched detections we will create new tracking in Kalman.
        # it means new object comes to the frame
        for i in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[i]))
            
        # delete tracks without detection for max_age frames in a row
        self.trackers = [tracker for tracker in self.trackers if tracker.time_since_last_update <= self.max_age]

        return out