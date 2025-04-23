# -*- coding = utf-8 -*-
# @Time:2022/10/19 10:40
# @Author : ZHANGTONG
# @File:utils.py
# @Software:PyCharm

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=0, batch_size=8, *args, **kwargs):
        if transform is None:
            transform = mytransforms
        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )