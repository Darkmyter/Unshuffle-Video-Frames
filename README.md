
<div align="center">

# Unshuffle Video Frames

Recreate the original of a corrupt video (the frames have been shuffled)

| Corrupted video | Unshuffled video |
|:--:|:--:|
| <img width="75%" src="resources/corrupted.gif"> | <img width="75%" src="resources/unshuffled.gif"> |
| <img width="75%" src="resources/theoffice-corrupted.gif"> | <img width="75%" src="resources/theoffice-unshuffled.gif"> |
| <img width="75%" src="resources/theoffice2-corrupted.gif"> | <img width="75%" src="resources/theoffice2-unshuffled.gif"> |
| <img width="75%" src="resources/corrupted-mall.gif"> | <img width="75%" src="resources/mall-unshuffled.gif"> |



</div>


## Installation

Create a virtual environment and install dependencies with `env.yaml` or `requirement.txt`.

## Usage

To unshuffle a video run the following command:

```
python unsuffle-video.py --path <corrupt-video-path> --output-path <output-path-video>
```

The unshuffled video is saved along with a flipped version.

For an extensive use of the tool, multiple parameters are available:

```
python unsuffle-video.py --help
Options:
  --path TEXT                                   Path to the video.
  --output-path TEXT                            Path to the output video.
  --yolo [yolov8s|yolov8m|yolov8l|yolov8x]      yolo network size.
  --similarity-subject [frame|object]           Similarity measure subject: all the frame or the objects                  
  --window INTEGER                              number of frame to sort at each iteration.
  --sort-by [iou|similarity]                    Metric used for selecting candidate frames at each iteration.
  --filter-by [iou|similarity]                  Metric used for filtering intruder images.
                                  
  --help                                        Show this message and exit.
```

<!-- Tips:
* Sorting by iou is more efficient is the camera moves slowly and and the path of the objects in the video is linear. 
* Sorting by similarity is beneficial when the camera angle change is important. -->


## How does it work?

The main ideas of the algorithm are to detect the objects (humans in our case) in the frames, find the frames that share high IoU and similarity (based on image embeddings) and sort those by object distance to recreate the trajectory.

The models used for object detection and embedding are YOLOv8 and ResNet50 respectively. You can find notes and article to explain both models in these links: [YOLOv8](https://github.com/Darkmyter/AI-resources/blob/main/Resources/Deep%20Learining/Computer%20vision/object-detection-papers.md#yolov8-2023) and [ResNet50](https://github.com/Darkmyter/AI-resources/blob/main/Resources/Deep%20Learining/Computer%20vision/image-classification-papers.md#deep-residual-learning-for-image-recognition-resnet-2015)


These ideas that motivated the algorithm are:
* Recreate the trajectories of the humans in the video.
* Handle multiple humans in the frames, and thus identify them between each couple of frames.
* Utilize the scene representation to decouple frames in case of complex trajectory.
* Handle camera movement around the scene.


Bellow is a simplified pseudo algorithm:

1. Detect objects using YOLOv8
2. Filter objects to keep only humans: this removes noise from objects all over the scene.
3. Compute IoU matrix $M_{IoU}$ of size $n_{frames}*n_{frames}$, Cosine Similarity matrix $M_{cos}$ of size $n_{frames}*n_{frames}$ (based on frame or objects) and a mapping that pairs objects of different frames.
4. Filter the frames by:
   - IoU: frames that have no IoU higher than 0.8 are eliminated. 
   - Cosine similarity: frames that have shares a similarity higher than 0.8 with less than 10% of the frames are eliminated.
5. Run the main iteration loop until all the frames are sorted or maximum number of iter reached.   
   Init: $L=[f^*]$ unshuffled frames, $f^*=0$ the first frame to sort, $Q=False$ whether to add sorted batch to beginning of the $L$ or not, 
   1. $C =$ sorted frames by IoUs or similarity in relation to $f^*$
   2. $C =$ filtered $C$ by an IoU and similarity thresholds
   3. $C =$ select a number of $window$ frames that are not in $L$, from $C$.
   4. Sort the $C$ based on objects centers:
      1. compute the distance between objects in $C$ starting from $f^*$ for each permutation.
      2. average over the objects.
      3. select the permutation with the least mean distance.
   5. add $C$ to the start or end of $L$ (based on $Q$)
   6. if $len(C) =0$ or $Q$: $f^*=$ first item of $L$  and $Q=True$
   7. else:  $f^*=$ the last item of $L$
   8. iterate again


## Reproduce the examples above:

The videos shown in the example can be found [here](https://drive.google.com/drive/folders/1Mh7nhp5S9KiI7FH9wlScO9BLe5zWiifx?usp=share_link).  
* example 1:
  * `python unsuffle-video.py --path corrupted.mp4`
* example 2:
  * `python unsuffle-video.py --path corrupted-theoffice.mp4 --sort_by similarity`
* example 3:
  * `python unsuffle-video.py --path corrupted-theoffice2.mp4`
* example 4:
  * `python unsuffle-video.py --path corrupted-mall.mp4 --window 7 --similarity-subject object`

To shuffle your own videos, use the following command:  
`python shuffle-video.py --path <video-path> --output-path <corrupted-video-output-path>`