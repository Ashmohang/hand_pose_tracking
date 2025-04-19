# Hand and Pose Landmark Extraction from Video using MediaPipe

This project uses [MediaPipe](https://google.github.io/mediapipe/) to extract hand and pose landmarks from a video and saves the extracted coordinates into structured CSV files. It provides both visualization and data logging of keypoints frame-by-frame.

## Features

- Detects full body pose and hand landmarks from video input
- Visual overlay using OpenCV + MediaPipe
- Structured export of hand/pose data into timestamped CSV files
- Supports single and dual hand detection
- Handedness classification (left vs right)

## Installation

```bash
pip install opencv-python mediapipe pandas numpy
```

## Usage

1. Place your video inside the `sample_video/` folder
2. Update the path inside `hand_pose_tracking.py`
3. Run the script:

```bash
python src/hand_pose_tracking.py
```

CSV outputs will be saved in the working directory.

## Example Output

- `HandDataAtHH_MM.csv`: Real-world 3D hand landmarks + timestamps + handedness
- `landmarks_data.csv`: Pose and hand keypoints frame-wise

## Applications

- Human pose analysis
- Sports or rehabilitation monitoring
- Gesture recognition pre-processing

## Author

Ashwin Mohan  
MSc Diagnostics, Data and Digital Health  
University of Warwick
