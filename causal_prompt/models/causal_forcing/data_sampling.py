"""Epoch-aware data iteration with a resumable batch cursor."""

from itertools import islice

from torch.utils.data import DataLoader


def cycle_batches(loader: DataLoader, batches_consumed: int = 0):
    """Reshuffle each epoch and resume at the exact next dataset batch.

    The loader uses DistributedSampler, whose seed plus epoch determines the
    ordering. Keeping one fixed epoch aliases with the G/critic update ratio.
    """
    if batches_consumed < 0:
        raise ValueError("batches_consumed must be non-negative")
    if len(loader) == 0:
        raise ValueError("Training loader is empty; check dataset size and world size")
    epoch, offset = divmod(batches_consumed, len(loader))
    while True:
        loader.sampler.set_epoch(epoch)
        yield from islice(loader, offset, None)
        epoch += 1
        offset = 0
