from torch.utils.data import IterableDataset
import PIL
import csv
import torch
from io import BytesIO
import base64


class TsvDataset(IterableDataset):
    def __init__(self, path, transform=None, caption_only=False):
        self.f = open(path, "r")
        self.tsvreader = csv.reader(self.f, delimiter='\t')
        self.transform = transform
        self.caption_only = caption_only
        def callback_fn(image_base64, id, caption):
            try:
                img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64))).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, id, caption
            except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError):
                print("UnidentifiedImageError")
                return torch.zeros((3, 256, 256)), "not_a_image", "not_a_caption"
        self.callback_fn = callback_fn
    def __iter__(self):
        def get_next():
            if self.caption_only:
                for line in self.tsvreader:
                    yield self.callback_fn(torch.zeros((3, 256, 256)), line[0], line[1])
            else:
                for line in self.tsvreader:
                    yield self.callback_fn(line[3], line[0], line[2])
        return iter(get_next())
    def __del__(self):
        self.f.close()