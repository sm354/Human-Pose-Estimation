# Checkerboard-Dataset

This modules concerns the pipeline used to create the checkerboard dataset (or "Artificial dataset") using [Coco dataset (2014)](https://cocodataset.org/#download) and [checkerboard patterns](path to sample images).

## Installation

Download the Coco dataset using [Coco API](https://github.com/cocodataset/cocoapi)

```bash
git clone https://github.com/cocodataset/cocoapi.git
```

Run `main.py` to create checkerboard dataset - placing the checkerboard patterns on coco images using given annotations.

```
python3 main.py --patterns_path /path/to/Pattern/folder/ --annotations_path /path/to/annotations.json --images_path /path/to/cocoimages --dataset_path /path/to/save/checkerboard/dataset
```

## Usage

### data-augmentation.py

```
python data-augmentation.py
```

The script applies brightness change, contrast change and saturation change on the pattern images.

### image_zoomOut.py

* The patterns that are obtained after augmentation are placed in a 2 x 2 matrix and the same pattern is pasted at each index. 
* Then the resulting image is resized to the same size as that of the original pattern.