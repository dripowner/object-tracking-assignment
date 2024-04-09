def IoU(bbox1, bbox2) -> float:
    if bbox1 is None or bbox2 is None:
        return 1
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

