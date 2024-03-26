import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Union
from PIL import Image, ImageDraw, ImageFont


class CityscapesDetectionAPI:

    def __init__(self, root: Path, annFile: Path):
        self.root = root
        self.annFile = annFile

        if not self.annFile.exists():
            raise FileNotFoundError(f'Annotation file not found at {self.annFile}.')

        print(f'[INFO] loading annotations into memory from {self.annFile.name}...')
        with open(self.annFile, 'r') as f: self.annotations = json.load(f)
        print('done!')

        print('creating index...')
        self.categories = self.annotations['categories']
        for cat in self.categories: cat['hasInstances'] = bool(cat['hasInstances'])
        self.images = self.annotations['images']
        self.annotations = self.annotations['annotations']
        print('index created!')

    def __getitem__(self, image_id):
        return self.loadImgs(image_id)[0], self.loadAnn(image_id)
    
    def getImageIds(self) -> List[int]:
        return [img['id'] for img in self.images]

    def getCategory(self, category_id) -> Dict[str, any]:
        return next((cat for cat in self.categories if cat['id'] == category_id), None)
    
    def loadAnn(self, image_id: int) -> List[Dict[str, any]]:
        return [ann for ann in self.annotations if ann['image_id'] == image_id]

    def loadImgs(self, image_ids: Union[List[int], int]) -> List[Image.Image]:
        if isinstance(image_ids, int):
            image_ids = [image_ids]
        return [Image.open(self.root / img['file_name']) for img
                in list(filter(lambda img: img['id'] in image_ids, self.images))]
    
    def showAnns(self, image: Image.Image, anns: List[Dict[str, any]], 
                 include_noInstances: bool = False, copy: bool = False) -> Image.Image:
        image = image.copy() if copy else image
        draw = ImageDraw.Draw(image)
        
        for instance_ann in anns:
            category = self.getCategory(instance_ann['category_id'])
            if not category['hasInstances'] and not include_noInstances:
                continue

            color = tuple(np.random.randint(0, 255, 3))
            x, y, w, h = instance_ann['bbox']
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x, y), category['name'], fill='white', font=ImageFont.truetype("arial.ttf", size=20))

        return image