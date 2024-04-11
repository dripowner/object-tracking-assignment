from typing import List, Dict
from collections import Counter

def num_id_switches(predicted_ids: List) -> int:
    curr_track = None
    count = 0
    for pred in range(len(predicted_ids) - 1):
        if predicted_ids[pred] not in ["?", "not_detected"]:
            curr_track = predicted_ids[pred]
        if (predicted_ids[pred + 1] not in ["?", "not_detected"] and 
            predicted_ids[pred + 1] != curr_track and
            curr_track is not None):
            count += 1
    return count

def max_track_len(predicted_ids: List) -> int:
    l = 0
    most_common_track_ids = Counter(predicted_ids).most_common()
    for track_id, track_id_count in most_common_track_ids:
        if track_id == '?' or track_id == 'not_detected':
            continue
        else:
            l = max(l, track_id_count)
    return l

def misses_count(predicted_ids: List) -> int:
    counts = dict(Counter(predicted_ids).most_common())
    return counts["not_detected"] + counts["?"]

def MOTA(predicted_ids_dict: Dict):
    fn = 0
    fp = 0
    idsw = 0
    gt_count = 0
    cb_ids = list(predicted_ids_dict.keys())
    for cb_id in cb_ids:
        fp += predicted_ids_dict[cb_id].count("?") # less than IoU threshold
        fn += predicted_ids_dict[cb_id].count("not_detected") # not detected
        idsw += num_id_switches(predicted_ids_dict[cb_id]) # times of track_id switches
        gt_count += len(predicted_ids_dict[cb_id])
        
    return 1 - (fn + fp + idsw) / (gt_count)


