
# Dataset Setup

This document explains how to prepare all datasets used in the **CDFSS** and **COCO** evaluation pipelines for this repository.


## 1. CDFSS Dataset Setup

The CDFSS evaluation uses the same dataset preprocessed by this repo: https://github.com/Vision-Kek/ABCDFSS/tree/main
Below are the datasets already processed by the referenced repo:

### Download Links

* **DeepGlobe**
  [https://www.kaggle.com/datasets/heyoujue/deepglobe](https://www.kaggle.com/datasets/heyoujue/deepglobe)
* **ISIC 2018 (class-wise)**
  [https://www.kaggle.com/datasets/heyoujue/isic2018-classwise](https://www.kaggle.com/datasets/heyoujue/isic2018-classwise)
* **SUIM (merged)**
  [https://www.kaggle.com/datasets/heyoujue/suim-merged](https://www.kaggle.com/datasets/heyoujue/suim-merged)

### Steps

1. Download and extract each dataset.
2. Update the `datapath` field in your configuration file (under the `configs/` folder) to point to the root directory containing the extracted dataset.

---

## 2. COCO Dataset Setup

For COCO subset preparation, we follow the procedure from:
[https://github.com/pasqualedem/LabelAnything](https://github.com/pasqualedem/LabelAnything)

### Step 1 — Download COCO 2017 Images and 2014 Annotations

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
```

### Step 2 — Extract and Organize

```bash
unzip "*.zip" && rm *.zip

mv val2017/* train2017/
mv train2017 train_val_2017
rm -rf val2017
```

### Step 3 — Normalize COCO 2014 Filenames

The COCO20i split requires renaming the 2014 annotation file entries so they match the 2017 image folder.
Run the script below:

```python
import json

def rename_coco20i_json(instances_path: str):
    """Normalize image filenames in COCO 2014 instance annotations."""
    with open(instances_path, "r") as f:
        anns = json.load(f)

    for image in anns["images"]:
        image["file_name"] = image["file_name"].split("_")[-1]

    with open(instances_path, "w") as f:
        json.dump(anns, f)

rename_coco20i_json('annotations/instances_train2014.json')
rename_coco20i_json('annotations/instances_val2014.json')
```

### Step 4 — Update Config

In your config file, update the fields:

* `instances_path` → path to the modified `instances_val2014.json`
* `img_dir` → path to the merged `train_val_2017` directory

---