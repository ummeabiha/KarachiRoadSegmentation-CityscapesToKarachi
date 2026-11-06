# Karachi Road Segmentation: Cityscapes to Karachi

Semantic, instance and panoptic segmentation of Karachi road photos using models trained on Cityscapes. The models are tested first as they are and then after fine-tuning on a small set of hand-labelled Karachi photos with a new `rickshaw` and `motorcycle` class.

Notebook: `KarachiRoadSegmentation.ipynb`

## Overview

| Type | Output | Model |
|---|---|---|
| Semantic | a class for every pixel | SegFormer-B2 (Cityscapes) |
| Instance | a separate mask for each object | Mask2Former Swin-L (Cityscapes panoptic), object part |
| Panoptic | a class for every pixel plus an ID for each object | Mask2Former Swin-L (Cityscapes panoptic) |

The notebook runs three stages:
1. **Cityscapes:** the models on 8 sample Cityscapes images (Aachen).
2. **Karachi without fine-tuning:** the original models on Karachi photos.
3. **Karachi with fine-tuning:** SegFormer fine-tuned on Karachi photos with a 20th class, `rickshaw`. Mask2Former masks are relabelled using the fine-tuned semantic map.

## Data

- **Cityscapes sample:** `cityscapes8` by Ultralytics, downloaded automatically.
- **Karachi photos:** phone photos and openly licensed web photos (Pexels, Wikimedia Commons). Commons credits are saved to `credits.csv`.
- **Annotation:** 11 photos labelled by hand in makesense.ai with polygons, exported as COCO JSON (`labels_coco.json`).
  - **Classes:** road, person, car, motorcycle, rickshaw, bus, truck, plus `ignore`.
  - **Objects:** 108 in total: 38 motorcycles, 25 rickshaws, 21 people, 17 cars, 4 trucks and 3 buses.
  - **Partial labels:** one traffic-jam photo is only partly labelled, so only its outlined areas are scored.
- **Split:** 7 training photos and 4 test photos. Near-duplicate photos stay on the same side.

## Results (4 test photos)

| Metric | Without fine-tuning | With fine-tuning | Change |
|---|---|---|---|
| Semantic mIoU | 39.1 | 52.9 | +13.8 |
| Pixel accuracy | 74.1 | 81.9 | +7.8 |
| Instance precision | 38.6 | 58.6 | +20.0 |
| Instance recall | 40.3 | 59.0 | +18.7 |
| Panoptic PQ | 30.8 | 46.5 | +15.7 |
| Panoptic SQ | 64.6 | 76.0 | +11.4 |
| Panoptic RQ | 38.6 | 59.4 | +20.8 |

Averages use the 6 classes with labelled objects in the test photos, so bus is left out.

**Semantic IoU per class (%)**

| Class | Without | With |
|---|---|---|
| road | 88.7 | 91.7 |
| person | 47.2 | 53.5 |
| car | 40.0 | 67.7 |
| motorcycle | 39.4 | 57.6 |
| rickshaw | 0.0 | 32.4 |
| truck | 19.5 | 14.6 |

## Findings

- **Road transfers well** from Cityscapes to Karachi. Vehicles don't, especially rickshaws (no class) and motorbikes (rare in Cityscapes).
- **Fine-tuning helps most for semantic segmentation:** rickshaws become recognisable, and motorbikes are no longer called bicycles.
- **Partial transfer:** instance and panoptic results improve through better classes, but most masks still come from Mask2Former trained on Cityscapes.
- **Remaining errors:** decorated loaders are called rickshaws, hazy sky is predicted as road, and rickshaws are over-predicted.

## Limitations

- Only 11 labelled photos and 4 test photos, so the scores are a rough guide.
- Too few buses to score.
- Mask2Former is not retrained.

## Privacy

- EXIF data, including GPS, is removed when photos are loaded.
- People found by the labels or the models are blurred before any image is shown. Number plates are not blurred.
- The photos are used only for coursework.

## How to run

1. Open the notebook in Google Colab with a T4 GPU.
2. Put the photos and `labels_coco.json` in this Drive folder:
   `MyDrive/Colab Notebooks/Advanced AI/Karachi Road Segmentation/Road Images`
3. Run all cells. Set `RETRAIN = True` to fine-tune again, for example after changing the labels. Otherwise the saved model in Drive is loaded.

**Main libraries:** PyTorch, Hugging Face Transformers, OpenCV, pandas and matplotlib.

## References

- Cordts et al., *The Cityscapes Dataset for Semantic Urban Scene Understanding*, CVPR 2016.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, NeurIPS 2021.
- Cheng et al., *Masked-attention Mask Transformer for Universal Image Segmentation*, CVPR 2022.
- Kirillov et al., *Panoptic Segmentation*, CVPR 2019.
