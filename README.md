## Introduction

lightningOCR is an open-source toolbox based on PyTorch and Lightning-AI for text recognition.

## Installation

> TODO: 

## Examples

| Example                                              | Description                                          |
| ---------------------------------------------------- | ---------------------------------------------------- |
| [Create_dataset](examples/0_create_dataset.ipynb)    | Create a primitive dataset from scratch              |
| [Train ABINet Vision](examples/1_train_abinet.ipynb) | Train the ABINet Vision model on the created dataset |
| [Export ONNX](examples/2_export_abinet_onnx.ipynb)   | Export the model to onnx format                      |

## [Model Zoo](./lightning_ocr/models)

- [x] [ABINet](./lightning_ocr/models/abinet.py) (CVPR 2021) from [MMOCR](https://github.com/open-mmlab/mmocr/blob/main/configs/textrecog/abinet/README.md)  
- [x] [MGP-STR](./lightning_ocr/models/mgp_str.py) (ECCV 2022) from [transformers](https://huggingface.co/docs/transformers/model_doc/mgp-str)  
- [x] [TrOCR](./lightning_ocr/models/trocr.py) from [transformers](https://huggingface.co/docs/transformers/model_doc/trocr)  

## Citation

```
@article{lightningOCR,
  title   = {{lightningOCR}: Open-source toolbox based on PyTorch and Lightning-AI for text recognition},
  author  = {MiXaiLL76},
  year    = {2024}
}
```
