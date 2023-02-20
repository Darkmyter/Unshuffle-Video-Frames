
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
                                  
  --help                                        Show this message and exit.
```

<!-- Tips:
* Sorting by iou is more efficient is the camera moves slowly and and the path of the objects in the video is linear. 
* Sorting by similarity is beneficial when the camera angle change is important. -->


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
