import numpy as np
import scipy
from typing import List

def IoU(bbox1, bbox2) -> float:
    if bbox1 is None or bbox2 is None:
        return 0
    # Coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    iou = inter_area / float(bbox1_area + bbox2_area - inter_area)

    return iou

class HungarianTracker():
    def __init__(self, iou_threshold) -> None:
        self.tracks_ids_dict = {}
        self.threshold = iou_threshold

    
    def track(self, bbox_list: List) -> List:
        """
        Return the dictionary of current tracks
        return: dict = {track_id: last_bbox}
        """
        result_list = []
        # On start (if dict is empty)
        if not bool(self.tracks_ids_dict):
            for track_id, bbox in enumerate(bbox_list):
                self.tracks_ids_dict[track_id] = {"bbox": bbox}
                result_list.append(track_id)
            return result_list
        else:
            # if we have more bboxes than tracks, create new ones
            if len(self.tracks_ids_dict.keys()) < len(bbox_list):
                for _ in range(len(bbox_list) - len(self.tracks_ids_dict.keys())):
                    self.tracks_ids_dict[max(self.tracks_ids_dict)+1]={"bbox": None}
            
            
            cost_matrix = np.zeros((len(self.tracks_ids_dict.keys()), len(bbox_list)))
            
            for track_id in list(self.tracks_ids_dict.keys()):
                for k in range(len(bbox_list)):
                    # Compute the IoU between last bbox for track_id and all current bboxes
                    iou = IoU(self.tracks_ids_dict[track_id]["bbox"], bbox_list[k])
                    cost_matrix[track_id][k] = 1 - iou
            
            original_cost_matrix = np.copy(cost_matrix)
            assignment = scipy.optimize.linear_sum_assignment(cost_matrix)

            
            # Assign the track_ids to the objects
            for bbox_id in range(len(bbox_list)):
                bbox = bbox_list[bbox_id]
                track_id = list(assignment[1]).index(bbox_id)
                if original_cost_matrix[track_id][bbox_id] > self.threshold:
                    result_list.append("?")
                else:
                    result_list.append(track_id)
                    self.tracks_ids_dict[track_id]['bbox'] = bbox
            print(original_cost_matrix)
            return result_list