from scipy.spatial import distance
from itertools import permutations
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import cv2


def read_video(video_path):
    video = cv2.VideoCapture(str(video_path))

    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    return frames

def get_video_writer(output_video_path, fps, width, height):
  """Create a video writer to save new frames after annotation"""
  return cv2.VideoWriter(
      str(output_video_path),
      fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
      fps=fps,
      frameSize=(width, height),
      isColor=True
  )

def save_video(frame_ids, frames, output_video_path):
    """Save video based on frame_ids order."""

    video_writer = get_video_writer(
        output_video_path,
        30,
        frames[0].shape[1],
        frames[0].shape[0]
    )

    for i in frame_ids:

        # save the frame to video writer
        video_writer.write(frames[i])

    # save the video
    video_writer.release()
    

def iou(bbox1, bbox2):
    """Compute IoU"""
    x1 = max(bbox1[0], bbox2[0])
    x2 = min(bbox1[2], bbox2[2])
    y1 = max(bbox1[1], bbox2[1])
    y2 = min(bbox1[3], bbox2[3])

    inter = max(0, x2 - x1) * max(0, y2 -y1)

    bbox1_area = (bbox1[2] - bbox1[0])*(bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0])*(bbox2[3] - bbox2[1])


    return inter / (bbox1_area + bbox2_area - inter)

def box_center(bbox):
    """Get bbox center."""
    return ((bbox[2] + bbox[0]) / 2, (bbox[1] + bbox[3]) / 2)

def bbox_distance(bbox1, bbox2):
    """Compute distance between two bboxes."""
    c1 = box_center(bbox1)
    c2 = box_center(bbox2)

    return distance.euclidean(c1, c2)


def get_embedding(frames, predictions=None):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    layer = model._modules.get('avgpool')

    outputs = []
    def copy_embeddings(model, input, output):
        output = output[:, :, 0, 0].detach().numpy().tolist()

        outputs.append(output)

    _ = layer.register_forward_hook(copy_embeddings)

    model.eval()

    tc = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])


    if predictions is not None:
        # crop objects
        embeddings = []
        for preds, frame in zip(predictions, frames):
            embeddings.append([])
            img_frame = Image.fromarray(frame)
            for box in preds:
                # print(box)
                img_obj = img_frame.crop(box)
                img_obj = tc(img_obj)
                img_obj = img_obj.unsqueeze(0)
                _ = model(img_obj)
                embeddings[-1].append(np.array(outputs[-1][0]))
    else:
        embeddings = []
        for frame in frames:
            img_frame = Image.fromarray(frame)
            img_frame = tc(img_frame)
            img_frame = img_frame.unsqueeze(0)
            _ = model(img_frame)
            embeddings.append(np.array(outputs[-1][0]))

    return embeddings

def cosine_similarity(emb1, emb2):

    cos_sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))

    return cos_sim

def sort_candidates(candidates, f_star_x, centers):
    """
    Sort the candidates based on the distance between the frames.
    
    The goal is to find the best order of candidates, where the objects trace the least distance from the f* to the farthest frame.
    In order to take into account all the objects and their different trajectories, we compute a distance instead of sorting directly the x coordinates.

    For each object, calculate the distance from one frame to the other and sum them. Then sum over all objects.

    The permutation with the least overall distance is selected.
    
    """
    dists = []

    # permutation of the frames.
    perm = list(permutations(range(len(candidates))))

    for p in perm:
        dists.append(0)
        for obj_centers, obj_x in zip(centers, f_star_x):
            # sort centers by perm and filter
            perm_centers = [obj_centers[j] for j in p if obj_centers[j] is not None]
            if perm_centers:
                # compute distance starting from f*.
                d = np.abs(perm_centers[0] - obj_x)
                for i in range(len(perm_centers) - 1):
                    d += np.abs(perm_centers[i] - perm_centers[i+1])
                dists[-1] += d

    if dists:
        # return the best permutation.
        return [candidates[j] for j in perm[np.argmin(dists)]]
    else:
        return []