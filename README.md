# Karachi Road Segmentation: Cityscapes to Karachi

How well do segmentation models trained on European streets work on Karachi roads? This project tests semantic, instance and panoptic segmentation models trained on Cityscapes on Karachi road photos. It tests them first as they are, then after fine-tuning on a few hand-labelled Karachi photos with a new `rickshaw` class.

- **Notebook:** [`KarachiRoadSegmentation.ipynb`](KarachiRoadSegmentation.ipynb)
- **Fine-tuned model:** [ummeabiha/segformer-b2-karachi-rickshaw](https://huggingface.co/ummeabiha/segformer-b2-karachi-rickshaw) on Hugging Face

## Contents

- [Overview](#overview)
- [Results](#results)
- [Findings](#findings)
- [Data and annotation](#data-and-annotation)
- [How to run](#how-to-run)
- [Repository contents](#repository-contents)
- [Limitations](#limitations)
- [Privacy](#privacy)
- [Licences](#licences)
- [References](#references)

## Overview

| Type | Output | Model |
|---|---|---|
| Semantic | a class for every pixel | SegFormer-B2 (Cityscapes) |
| Instance | a separate mask for each object | Mask2Former Swin-L (Cityscapes panoptic), object part |
| Panoptic | a class for every pixel plus an ID for each object | Mask2Former Swin-L (Cityscapes panoptic) |

The notebook runs three stages:
1. **Cityscapes:** the models on 8 sample Cityscapes images from Aachen.
2. **Karachi without fine-tuning:** the original models on Karachi photos.
3. **Karachi with fine-tuning:** SegFormer fine-tuned on Karachi photos with a 20th class, `rickshaw`.

For instance and panoptic segmentation, Mask2Former is not retrained. Its object masks take their classes from the fine-tuned semantic map, and rickshaws and motorbikes that Mask2Former misses are added from that map.

## Results

All scores are on 4 held-out Karachi test photos.

| Metric | Without fine-tuning | With fine-tuning | Change |
|---|---|---|---|
| Semantic mIoU | 39.1 | 52.9 | +13.8 |
| Pixel accuracy | 74.1 | 81.9 | +7.8 |
| Instance precision | 38.6 | 58.6 | +20.0 |
| Instance recall | 40.3 | 59.0 | +18.7 |
| Panoptic PQ | 30.8 | 46.5 | +15.7 |
| Panoptic SQ | 64.6 | 76.0 | +11.4 |
| Panoptic RQ | 38.6 | 59.4 | +20.8 |

Averages use the 6 classes with labelled objects in the test photos, so bus is left out. An object counts as found when its mask overlaps a labelled object with IoU > 0.5.

**Semantic IoU per class (%)**

| Class | Cityscapes sample | Karachi, without | Karachi, with |
|---|---|---|---|
| road | 98.9 | 88.7 | 91.7 |
| person | 66.8 | 47.2 | 53.5 |
| car | 91.6 | 40.0 | 67.7 |
| motorcycle | n/a | 39.4 | 57.6 |
| rickshaw | n/a | 0.0 | 32.4 |
| truck | n/a | 19.5 | 14.6 |

Motorcycle and truck are marked n/a for Cityscapes because they cover only a few pixels in the 8 sample images. Cityscapes has no rickshaw class.

## Findings

- **Road transfers well** from Cityscapes to Karachi. Vehicles don't, especially rickshaws (no class) and motorbikes (rare in Cityscapes).
- **Without fine-tuning,** rickshaws are mostly called cars, and about half of the motorbike pixels are called bicycles.
- **Fine-tuning helps most for semantic segmentation:** rickshaws become recognisable, and motorbikes are no longer called bicycles.
- **Partial transfer:** instance and panoptic results improve through better classes, but most masks still come from Mask2Former trained on Cityscapes. Rickshaw recall rose from 0 to 60%, and motorbike recall from 17% to 50%.
- **Remaining errors:** decorated loaders are called rickshaws, hazy sky is predicted as road, and rickshaws are over-predicted.

## Data and annotation

- **Cityscapes sample:** `cityscapes8` by Ultralytics, downloaded automatically by the notebook.
- **Karachi photos:** phone photos and openly licensed web photos (Pexels, Wikimedia Commons).
- **Labels:** 11 photos labelled by hand in makesense.ai with polygons, exported as COCO JSON (`labels_coco.json`).
  - **Classes:** road, person, car, motorcycle, rickshaw, bus, truck, plus `ignore` for areas left out of scoring.
  - **Objects:** 108 in total: 38 motorcycles, 25 rickshaws, 21 people, 17 cars, 4 trucks and 3 buses.
  - **Partial labels:** one traffic-jam photo is only partly labelled, so only its outlined areas are scored.
- **Split:** 7 training photos and 4 test photos. Near-duplicate photos stay on the same side.
- **Unlabelled photos:** 16 more busy-road photos are compared by class share and object counts but not scored.

## How to run

1. Open the notebook in Google Colab and choose a T4 GPU (**Runtime > Change runtime type**).
2. Put the photos and `labels_coco.json` in this Google Drive folder:
   ```
   MyDrive/Colab Notebooks/Advanced AI/Karachi Road Segmentation/Road Images
   ```
3. In section 1, set `HF_MODEL` to the Hugging Face model name.
4. Run all cells.

**Where the fine-tuned model comes from:** the notebook loads it from `Outputs/` in Drive, or from Hugging Face if it isn't in Drive. It trains a new model only if neither has one, or if `RETRAIN = True`. Retrain after changing the labels.

**Main libraries:** PyTorch, Hugging Face Transformers, OpenCV, pandas and matplotlib. The notebook installs what Colab is missing.

## Repository contents

| File | Description |
|---|---|
| `KarachiRoadSegmentation.ipynb` | the full experiment, with saved outputs |
| `KarachiRoadSegmentation` | all road and common images, along with hand labels for the 11 Karachi photos |
| `README.md` | this file |

The fine-tuned model is on Hugging Face.

## Limitations

- Only 11 labelled photos and 4 test photos, so one object more or less can change a class score by 15-50 points. Treat the scores as a rough guide.
- There are too few buses to score.
- Mask2Former is not retrained, so missed or merged object masks stay wrong.
- Sky, buildings and trees were not hand-labelled.

## Privacy

- Photos were taken in public places, of traffic in general, with no close-ups of people.
- EXIF data, including GPS, is removed when photos are loaded.
- People found by the labels or the models are blurred before any image is shown. Number plates are not blurred.
- The photos are used only for coursework.

## Licences

- **Fine-tuned model:** a derivative of NVIDIA's SegFormer, under the [NVIDIA Source Code License for SegFormer](https://github.com/NVlabs/SegFormer/blob/master/LICENSE). It may be used only for non-commercial purposes, meaning research or evaluation.
- **Pretrained models and datasets:** SegFormer, Mask2Former and Cityscapes are used under their own licences.
- **Web photos:** these keep their original licences. Commons credits are saved to `credits.csv`.

## References

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR 2022.
- Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
