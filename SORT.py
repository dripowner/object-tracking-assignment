from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
from utils import IoU
from typing import List

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  
def compare_boxes(detections,trackers,iou_thresh=0.3):

    iou_matrix = np.zeros(shape=(len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = IoU(det,trk)
    
    # calculate maximum iou for each pair through hungarian algorithm
    row_id, col_id = linear_sum_assignment(-iou_matrix)
    matched_indices = np.transpose(np.array([row_id,col_id]))
    # geting matched ious
    iou_values = np.array([iou_matrix[row_id,col_id] for row_id,col_id in matched_indices])
    best_indices = matched_indices[iou_values > iou_thresh]

    unmatched_detection_indices = np.array([d for d in range(len(detections)) if d not in best_indices[:,0]])  
    unmatched_trackers_indices = np.array([t for t in range(len(trackers)) if t not in best_indices[:,1]])

    return best_indices,unmatched_detection_indices,unmatched_trackers_indices

class KalmanBoxTracker():
    count = 0 # id of track
    def __init__(self, bbox) -> None:
        # x - state, z - measurement
        # x: [x, y, r, s, x', y', r'], z: [x, y, r, s]
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # from x to z
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        # from x(k) to x(k+1)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        # covariance matrix of measurements noises
        self.kf.R[2:,2:] *= 10.
        # covariance matrix of state (uncertainty in state)
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        # covariance matrix of model error (consider that model works good enough)
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.time_since_last_update = 0
    
    def update(self,bbox):
        """
        Add new measurement z from detection bbox
        """
        # add new measurement 
        self.history = []
        self.time_since_last_update = 0
        self.kf.update(convert_bbox_to_z(bbox))
    
    def predict(self):
        """
        Predict new state x and return predicted bbox
        """
        # calculate new state x
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.history.append(convert_x_to_bbox(self.kf.x))
        self.time_since_last_update += 1
        return self.history[-1]
    
    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class SORT():
    def __init__(self, threshold=0.3, max_age=3) -> None:
        self.trackers = []
        self.iou_threshold = threshold
        # amount of frames track live without matching
        self.max_age = max_age
    
    def update(self, dets: List[List]) -> None:
        # get tracks prediction bboxes
        self.trackers = [tracker for tracker in self.trackers if not np.any(np.isnan(tracker.predict()))]

        trks = [tracker.get_state()[0].tolist() for tracker in self.trackers]

        # compare detection and tracks predictions based on IOU
        matched, unmatched_dets, unmatched_tracks = compare_boxes(dets, trks, iou_thresh=self.iou_threshold)

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


