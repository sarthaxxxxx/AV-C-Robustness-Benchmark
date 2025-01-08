# EPIC KITCHENS Dataset - Video Release
Release Date: April 2018

## Authors
Dima Damen (1)
Hazel Doughty (1)
Sanja Fidler (2)
Antonino Furnari (3)
Evangelos Kazakos (1)
Giovanni Maria Farinella (3)
Davide Moltisanti (1)
Jonathan Munro (1)
Toby Perrett (1)
Will Price (1)
Michael Wray (1)

* (1 University of Bristol)
* (2 University of Toronto)
* (3 University of Catania)


## Citing
When using the dataset, kindly reference:

Dima Damen, Hazel Doughty, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Giovanni Maria Farinella, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, Michael Wray (2018). Scaling Egocentric Vision: The EPIC-KITCHENS Dataset. check publication [here](http://epic-kitchens.github.io)

## Dataset Details
This readme contains information about video files.  Please see the 
[github](https://github.com/epic-kitchens/annotations) for the latest annotations 
and challenges for the dataset.

## Folder Details
We have the following top level folders:

* **videos:** contains the raw uncut videos.
* **object_detection_images:** contains the frames used as input for object
  detection.
* **rgb_flow:** contains the RGB and flow frames used as input for action
  recognition.

We use `P##/P##_**` with `##` denoting the participant number and `**` to denote
the video number.

### Video Folder Structure
Videos is further broken down into two subdirectories, train and test containing
the files in the train and test sets respectively.

Each video, labelled as `P##_**.mp4` can be found inside the participant
directory, i.e. `P01_01.mp4`  can be found at `Videos/train/P01/P01_01.mp4`.


### Object Images Folder Structure
Object Detection Images is further broken down into two subdirectories, train
and test containing the files in the train and test sets respectively.

Image frames, labelled as `img_xxxxxxxxxx.jpg` can be found inside the video
directory, i.e. The first frame from `P01_01.mp4` can be found at
`ObjectDetectionImages/train/P01/P01_01/img_0000000001.jpg`.


### RGB Flow Folder Structure
RGB Flow is further split into two subdirectories, `rgb` and `flow` containing
files for the RGB frames and flow Frames.  Both folders contain the same
directory structure and are further broken down into train and test directories
containing the train and test sets respectively.

Frames from a video have been grouped into a tar file, labelled as `P##_**.tar`
and can be found inside the participant directory, i.e. The RGB frames from,
`P01_01` can be found in `rgb_flow/train/P01_01.tar`. Similarly the flow frames
can be found in `flow/train/P01_01.tar`. Each tar file contains a flat directory
of frames named `frame_xxxxxxxxxx.jpg`.


## Video Information

Videos are recorded in 1080p at 59.94 FPS on a GoPro Hero 5 with linear field of
view. There are few videos which were shot at different resolutions,
field of views, or FPS due to participant error or camera. These videos
identified using `ffprobe` are:

* 1280x720: `P12_01`, `P12_02`, `P12_03`, `P12_04`.
* 2560x1440: `P12_05`, `P12_06` 
* 29.97 FPS: `P09_07`, `P09_08`, `P10_01`, `P10_04`, `P11_01`, `P18_02`,
    `P18_03`
* 48 FPS: `P17_01`, `P17_02`, `P17_03`, `P17_04`
* 90 FPS: `P18_09`

The GoPro Hero 5 was also set to drop the framerate in low light conditions to
preserve exposure leading to variable FPS in some videos.  If you wish to
extract frames we suggest you resample at 60 FPS to mitigate issues with
variable FPS, this can be achieved in a single step with FFmpeg: 

```
ffmpeg -i 'P##_**.MP4' -vf 'scale=-2:256' -q:v 4 -r 60 'P##_**/frame_%010d.jpg'
```

Optical flow was extracted using a fork of
[`gpu_flow`](https://github.com/feichtenhofer/gpu_flow) made 
[available on github](https://github.com/dl-container-registry/furnari-flow).
 We set the parameters: stride = 2, dilation = 3, bound = 25 and size = 256.


## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommerial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.
