from ultralytics import YOLO
import numpy as np
import numpy as np
import click

from utils import read_video, get_embedding, cosine_similarity, iou, box_center, sort_candidates, save_video

import warnings
warnings.filterwarnings("ignore")


@click.command()
@click.option('--path', help='Path to the video.')
@click.option('--output-path', default=None, help='Path to the output video.')
@click.option('--yolo', default="yolov8s", type=click.Choice(['yolov8s', 'yolov8m', 'yolov8l', 'yolov8x'], case_sensitive=True),  help='yolo network')
@click.option('--emb-subject', default="frame", type=click.Choice(['frame', 'object'], case_sensitive=True),  help='Embedding subject')
@click.option('--window', default=3, help='window size.', type=int)
@click.option('--sort-by', default="iou", type=click.Choice(['iou', 'similarity'], case_sensitive=True),  help='Metric used for selecting candidate frames.')
def main(path, output_path, yolo, emb_subject, window, sort_by):

    # load video frames
    frames = read_video(path)

    click.echo(f"Loaded video. {len(frames)} frames found.")

    ####################### detect objects ######################
    click.echo(f"Running predictions with {yolo} ...")
    model = YOLO(yolo)
    predictions = model(frames, verbose=False)

    # filter prediciton by class: 0 -> human
    filtered_predictions = []
    for pred in predictions:
        bboxes = []
        for box in pred.boxes:
            if box.cls.item() == 0:
                bboxes.append([int(i) for i in box.xyxy.tolist()[0]])
        filtered_predictions.append(bboxes)

    ####### compute embedding of each frame / object ############
    click.echo(f"Computing embeddings for {emb_subject} ...")
    if emb_subject == "frame":
        embeddings = get_embedding(frames)
    else:
        embeddings = get_embedding(frames, filtered_predictions)

    #### Compute IOU, object paring and embedding matrixes ######

    click.echo(f"Creating matrixes...")
    iou_thr_pairing = 0.4

    object_pairs = []
    iou_matrix = []
    cosine_matrix = []

    # loop through frames (rows of the matrix)
    for i, f0_bboxes in enumerate(filtered_predictions):
        iou_matrix.append([])
        object_pairs.append([])
        cosine_matrix.append([])

        # loop through frames (cols of the matrix)
        for j, f1_bboxes in enumerate(filtered_predictions):

            # compute iou between bboxes from each frame
            iou_sub_matrix = np.array([[iou(bbox_0, bbox_1) for bbox_1 in f1_bboxes] for bbox_0 in f0_bboxes])

            if emb_subject == "frame":
                # compute cosine similarity between frames
                cosine_matrix[-1].append(cosine_similarity(embeddings[j], embeddings[i]))
            else:
                # compute cosine similarity between objects
                cosine_sub_matrix = [[cosine_similarity(emb0, emb1) for emb1 in embeddings[j]] for emb0 in embeddings[i]]

            if not iou_sub_matrix.size == 0:
                # get max iou for each objects in f0
                max_ious = np.array(iou_sub_matrix).max(axis=1)
                # set iou_matrix[f0, f1] = mean of objects iou
                iou_matrix[-1].append(max_ious.mean())

                # link objects in the frames by iou.
                boxes_argmax = iou_sub_matrix.argmax(axis=1)
                boxes_max = iou_sub_matrix.max(axis=1)
                object_pairs[-1].append([boxes_argmax[z] if boxes_max[z] > iou_thr_pairing else None for z in range(len(f0_bboxes))])

                if emb_subject == "object":
                    # set mean of max cosine of objects as cosine similarity between frames.
                    max_cosine_obj = np.array(cosine_sub_matrix).max(axis=1)
                    cosine_matrix[-1].append(max_cosine_obj.mean())
            else:
                iou_matrix[-1].append(0)
                object_pairs[-1].append([])
                if emb_subject == "object":
                    cosine_matrix[-1].append(0)

    iou_matrix = np.array(iou_matrix)
    cosine_matrix = np.array(cosine_matrix)


    ############# filter intruder images by iou ####################

    click.echo(f"Filtering frames...")
    filtered_frames_ids = []
    for f_id in range(len(frames)):
        if not np.where(iou_matrix[f_id]> 0.8)[0].size == 0:
            filtered_frames_ids.append(f_id)

    # filter lists and matrixes
    filtered_frames = [frames[f] for f in filtered_frames_ids]
    filtered_predictions = [filtered_predictions[f] for f in filtered_frames_ids]
    filtered_iou_matrix = iou_matrix[np.ix_(filtered_frames_ids, filtered_frames_ids)]
    filtered_cosine_matrix = cosine_matrix[np.ix_(filtered_frames_ids, filtered_frames_ids)]
    filtered_object_pairs = [[object_pairs[i][j] for j in filtered_frames_ids] for i in filtered_frames_ids]


    click.echo(f"Removed {len(frames) - len(filtered_frames)} intruder frames.")

    ######################### Main Algorithm ##########################

    # number of frame to sort at each iteration
    iou_thr = 0.4
    sim_thr = 0.85

    # start with frame* = 0
    f_star = 0
    unshuffled_frames = np.array([f_star], dtype=int)

    # iteration
    num_iter = 0
    max_iter = 100

    # add frames to start or end of  the queue
    add_to_start = False
    
    click.echo(f"Unshuffling video...")
    while len(unshuffled_frames) != len(filtered_frames) and num_iter < max_iter:

        # select best candidate frames by sorting on iou, where is it higher than iou_thr.
        # the filtering insures only close frames are selected.
        if sort_by == "iou":
            candidates = np.argsort(-filtered_iou_matrix[f_star])
        else:
            sort_by == "similarity":
            candidates = np.argsort(-filtered_cosine_matrix[f_star])

        candidates = candidates[np.where(filtered_iou_matrix[f_star, candidates] > iou_thr)[0]]
        candidates = candidates[np.where(filtered_cosine_matrix[f_star, candidates] > sim_thr)[0]]

        candidates = list(candidates)

        # remove frames already seen
        candidates = np.array([c for c in candidates if c not in unshuffled_frames][:window])

        # get center coordinates of each object in the candidate frames that is linked to an object in f*
        centers = []
        f_star_x = []
        for obj in range(len(filtered_predictions[f_star])):
            centers.append([])
            for f in candidates:
                # get linked object
                pair = filtered_object_pairs[f_star][f][obj]
                if pair is not None:
                    pair_box = filtered_predictions[f][pair]
                    centers[-1].append(box_center(pair_box)[0])
                else:
                    centers[-1].append(None)
            f_star_x.append(box_center(filtered_predictions[f_star][obj])[0])


        candidates = sort_candidates(candidates, f_star_x, centers)
        
        # add candidates
        if add_to_start:
            candidates = np.flip(candidates)
            unshuffled_frames = np.concatenate([candidates, unshuffled_frames])
        else:
            unshuffled_frames = np.concatenate([unshuffled_frames, candidates])

        # select last added frame and execute again
        if len(candidates) == 0 or add_to_start:
            f_star = int(unshuffled_frames[0])
            add_to_start = True
        else:
            f_star = int(unshuffled_frames[-1])

        num_iter += 1

    unshuffled_frames = unshuffled_frames.astype(int)
    

    ####################### Save video #############################

    click.echo(f"Saving video...")
    if output_path is None:
        output_path = path.split(".")[0] + "-unshuffled." + path.split(".")[-1]

    save_video(unshuffled_frames, filtered_frames, output_path)

    unshuffled_frames = np.flip(unshuffled_frames) 

    save_video(unshuffled_frames, filtered_frames, output_path.split(".")[0] + "-flipped." + output_path.split(".")[-1])


if __name__ == "__main__":
    main()