# from https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
import torch
import numpy as np

def cutout(batch, length, img):
    _, _, h, w = batch.size
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    y1 = np.clip(y - length // 2, 0, h)
    y2 = np.clip(y + length // 2, 0, h)
    x1 = np.clip(x - length // 2, 0, w)
    x2 = np.clip(x + length // 2, 0, w)
    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets

class DataPrefetcher():
    def __init__(self, loader, mean, std, cutout_impl):
        self.loader = loader
        self.iter = iter(loader)
        self.length = len(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([m * 255 for m in mean]).cuda().view(1, 3, 1, 1).float()
        self.std = torch.tensor([s * 255 for s in std]).cuda().view(1, 3, 1, 1).float()
        self.cutout = cutout_impl
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.iter)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # # more code for the alternative if record_stream() doesn't work:
            # # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            # TODO: cutout

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_input
        target = self.next_target
        if inputs is None or target is None:
            raise StopIteration
        inputs.record_stream(torch.cuda.current_stream())
        target.record_stream(torch.cuda.current_stream())
        self.preload()
        return inputs, target

    def __len__(self):
        return self.length

    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        return self

    def __next__(self):
        return self.next()