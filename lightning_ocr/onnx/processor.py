from enum import IntEnum
from typing import Optional, Dict, List, Union
import numpy as np
import PIL.Image
import json

class Resampling(IntEnum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1

class ImageProcessor():
    def __init__(
        self,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: Resampling = Resampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: Optional[bool] = None,
    ) -> None:
        size = size if size is not None else {"height": 224, "width": 224}
        
        if not isinstance(size, dict):
            if isinstance(size, int):
                size = {"height": size, "width": size}
            elif isinstance(size, (tuple, list)):
                size = {"height": size[0], "width": size[1]}
            else:
                raise ValueError(
                    f"Could not convert size input to size dict: {size}. Size must be an int, tuple or dict."
                )
            
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.size = size
        self.resample = resample
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb

    @classmethod
    def load_from_file(cls, file : str):
        with open(file, "r") as f:
            kwargs = json.load(f)
        
        if "image_processor_type" in kwargs:
            del kwargs["image_processor_type"]    

        if "processor_class" in kwargs:
            del kwargs["processor_class"] 
           
        return cls(**kwargs)

    @staticmethod
    def batch_images(images):
        if isinstance(images, PIL.Image.Image):
            return [images], False
        
        elif isinstance(images, np.ndarray) and len(images.shape) == 3:
            return [images], False
        
        elif isinstance(images, list) and len(images) > 0 and (isinstance(images[0], PIL.Image.Image) or isinstance(images[0], np.ndarray)):
            return images, True
        
        elif isinstance(images, np.ndarray) and len(images.shape) == 4:
            return images, True
        else:
            raise ValueError(
                "Images must be a PIL image, a numpy array, or a list of PIL images/numpy arrays."
            )
    
    @staticmethod
    def convert_to_rgb(image):
        if not isinstance(image, PIL.Image.Image):
            return image

        if image.mode == "RGB":
            return image

        image = image.convert("RGB")
        return image
    
    @staticmethod
    def to_numpy_array(img) -> np.ndarray:
        if isinstance(img, PIL.Image.Image):
            return np.array(img)
        return np.array(img)

    def resize(self, img):
        return np.array(PIL.Image.fromarray(img).resize((self.size["width"], self.size["height"]), resample=self.resample))

    def rescale(self, img):
        rescaled_image = img.astype(np.float64) * self.rescale_factor
        return rescaled_image
    
    def preprocess(self, images):
        images, is_batched = ImageProcessor.batch_images(images)

        if self.do_convert_rgb:
            images = [ImageProcessor.convert_to_rgb(image) for image in images]
        
        # All transformations expect numpy arrays.
        images = [ImageProcessor.to_numpy_array(image) for image in images]
        
        if self.do_resize:
            images = [self.resize(img=image) for image in images]
        
        if self.do_rescale:
            images = [self.rescale(img=image) for image in images]
        
        if self.do_normalize:
            images = [
                (image - np.array(self.image_mean) * 255) / (np.array(self.image_std) * 255)
                for image in images
            ]
        
        return [image.transpose(2, 0, 1).astype(np.float32) for image in images], is_batched