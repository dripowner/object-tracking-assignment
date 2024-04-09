import numpy as np
from typing import List
import scipy
from utils import IoU

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
            
            # matrix to store IOU between each pair of dets and every track's last det
            cost_matrix = np.zeros((len(self.tracks_ids_dict.keys()), len(bbox_list)))
            
            for track_id in list(self.tracks_ids_dict.keys()):
                for k in range(len(bbox_list)):
                    # Compute the IoU between last bbox for track_id and all current bboxes
                    iou = IoU(self.tracks_ids_dict[track_id]["bbox"], bbox_list[k])
                    cost_matrix[track_id][k] = 1 - iou
            
            # if some bbox have intersection with one of tracks we consider that
            # this bbox not belongs to new tracks
            for col in cost_matrix[:, range(cost_matrix.shape[1])]:
                if np.any(np.logical_or(col!= 1, col!= 0)):
                    col[col==0] = 1

            # hungarian algo
            original_cost_matrix = np.copy(cost_matrix)
            assignment = scipy.optimize.linear_sum_assignment(cost_matrix)

            
            # Assign the track_ids to the objects
            for bbox_id in range(len(bbox_list)):
                bbox = bbox_list[bbox_id]
                track_id = list(assignment[1]).index(bbox_id)
                # no intersection -> no track_id
                if original_cost_matrix[track_id][bbox_id] > self.threshold:
                    result_list.append("?")
                else:
                    result_list.append(track_id)
                    self.tracks_ids_dict[track_id]['bbox'] = bbox
            print(result_list)
            return result_list