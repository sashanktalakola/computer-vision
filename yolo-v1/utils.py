import torch

def non_max_suppression(predictions, conf_thresh=0.5, nms_thresh=0.4, max_boxes=50):
    """
    Perform Non-Maximum Suppression (NMS) on the predictions from the model.

    Parameters:
    - predictions: (batch_size, num_boxes, 6) - Bounding boxes, class scores, and confidence scores
      Each box is represented by (x, y, w, h, confidence, class_score).
    - conf_thresh: Confidence threshold to filter out weak predictions.
    - nms_thresh: IoU threshold for NMS. If two boxes have IoU > nms_thresh, one will be suppressed.
    - max_boxes: Maximum number of boxes to keep after NMS.
    
    Returns:
    - list of lists: Each list contains the remaining boxes after NMS, where each box is represented
      by (x, y, w, h, confidence, class_score).
    """
    all_boxes = []

    for batch_idx in range(predictions.size(0)):
        # Filter out boxes with low confidence
        conf_mask = predictions[batch_idx, :, 4] > conf_thresh
        boxes = predictions[batch_idx, conf_mask]
        
        if boxes.size(0) == 0:
            continue
        
        # Extract boxes and scores
        box_coords = boxes[:, :4]  # (x, y, w, h)
        confidences = boxes[:, 4]  # confidence score
        class_scores = boxes[:, 5:]  # class scores for each class
        
        # Calculate IoU and perform NMS for each class separately
        for class_idx in range(class_scores.size(1)):
            # Get the class scores for the current class
            class_probs = class_scores[:, class_idx]
            
            # Sort by confidence score * class probability
            score = confidences * class_probs
            sorted_indices = score.argsort(descending=True)
            
            keep = []
            while sorted_indices.numel() > 0:
                # Select the box with the highest score
                idx = sorted_indices[0]
                keep.append(idx.item())
                
                if len(keep) >= max_boxes:
                    break
                
                # Get the current box's coordinates
                curr_box = box_coords[idx].unsqueeze(0)
                
                # Calculate IoU with all other boxes
                rest_boxes = box_coords[sorted_indices[1:]]
                iou = compute_iou(curr_box, rest_boxes)
                
                # Keep boxes with IoU less than the threshold
                sorted_indices = sorted_indices[1:][iou < nms_thresh]

            # Keep the selected boxes
            all_boxes.append(boxes[keep])
    
    return all_boxes


def compute_iou(boxes1, boxes2):
    """
    Compute the Intersection over Union (IoU) between two sets of boxes.
    boxes1: (N, 4) - N bounding boxes in the format (x, y, w, h)
    boxes2: (M, 4) - M bounding boxes in the format (x, y, w, h)
    
    Returns:
    - iou: (N, M) IoU matrix
    """
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    boxes1_x1 = boxes1[:, 0] - boxes1[:, 2] / 2
    boxes1_y1 = boxes1[:, 1] - boxes1[:, 3] / 2
    boxes1_x2 = boxes1[:, 0] + boxes1[:, 2] / 2
    boxes1_y2 = boxes1[:, 1] + boxes1[:, 3] / 2
    
    boxes2_x1 = boxes2[:, 0] - boxes2[:, 2] / 2
    boxes2_y1 = boxes2[:, 1] - boxes2[:, 3] / 2
    boxes2_x2 = boxes2[:, 0] + boxes2[:, 2] / 2
    boxes2_y2 = boxes2[:, 1] + boxes2[:, 3] / 2

    # Calculate intersection area
    inter_x1 = torch.max(boxes1_x1, boxes2_x1)
    inter_y1 = torch.max(boxes1_y1, boxes2_y1)
    inter_x2 = torch.min(boxes1_x2, boxes2_x2)
    inter_y2 = torch.min(boxes1_y2, boxes2_y2)
    
    inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_width * inter_height

    # Calculate union area
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union = boxes1_area + boxes2_area - intersection

    # Compute IoU
    iou = intersection / union
    return iou


def yolo_loss(predictions, targets, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5, lambda_obj=1, lambda_class=1):
    """
    predictions: (batch_size, S, S, 5 * B + C)
    targets: (batch_size, S, S, 5 * B + C)
    """
    batch_size = predictions.size(0)
    grid_size = S
    
    # Split predictions and targets
    pred_box = predictions[..., :B*5].reshape(batch_size, grid_size, grid_size, B, 5)
    pred_class = predictions[..., B*5:]
    
    target_box = targets[..., :B*5].reshape(batch_size, grid_size, grid_size, B, 5)
    target_class = targets[..., B*5:]

    # Initialize loss components
    loss_coord = 0
    loss_conf_obj = 0
    loss_conf_noobj = 0
    loss_class = 0

    # Loop over the batch
    for b in range(batch_size):
        for i in range(grid_size):
            for j in range(grid_size):
                # Check if there's an object in this grid cell
                obj_mask = target_box[b, i, j, :, 4] > 0
                noobj_mask = ~obj_mask

                # For object cells (IoU-based confidence and class loss)
                for k in range(B):
                    if obj_mask[k]:
                        # Coordinate loss (x, y, width, height)
                        loss_coord += lambda_coord * ((pred_box[b, i, j, k, 0] - target_box[b, i, j, k, 0])**2 +
                                                      (pred_box[b, i, j, k, 1] - target_box[b, i, j, k, 1])**2 +
                                                      (pred_box[b, i, j, k, 2] - target_box[b, i, j, k, 2])**2 +
                                                      (pred_box[b, i, j, k, 3] - target_box[b, i, j, k, 3])**2)

                        # Confidence loss (objectness)
                        loss_conf_obj += lambda_obj * (pred_box[b, i, j, k, 4] - target_box[b, i, j, k, 4])**2

                        # Class loss (class probabilities)
                        loss_class += lambda_class * torch.sum((pred_class[b, i, j, :] - target_class[b, i, j, :])**2)

                # For no-object cells (confidence loss)
                for k in range(B):
                    if noobj_mask[k]:
                        # Confidence loss for no-object cells
                        loss_conf_noobj += lambda_noobj * (pred_box[b, i, j, k, 4] - target_box[b, i, j, k, 4])**2

    # Average loss over the batch
    total_loss = (loss_coord + loss_conf_obj + loss_conf_noobj + loss_class) / batch_size
    return total_loss