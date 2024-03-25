import torch


class CacheInterface:
    # cache for ONE layer

    def update(self, key, value):
        """Update the cache for this key and value.
        
        The key, and val will have shape (Batch, Heads, Seqlen, Head dim)
        The cache is free to store them in a different format.
        Return the full cache after update.
        This cache instance need to know which position / layer is 
        the update for.
        """

class KVCachePrefill:

    def __init__(self, kv_quantize=False):
        self.kv_quantize = kv_quantize 
        self.cache_k = None
        self.cache_v = None

    def update(self, key, value):
        """This cache just remembers the stuff."""
        self.cache_k = key
        self.cache_v = value
        return key, value

    def state(self):
        return self.cache_k, self.cache_v



# Refactor out cache management
# Easier to test for quantized kv cache
class KVCacheGenerate:

    def __init__(self, 
        cache_k: torch.Tensor,  # previous cache
        cache_v: torch.Tensor,  # previous cache
        position: int,  # position to store the cache
    ):
        super().__init__()
        self.cache_k = cache_k
        self.cache_v = cache_v
        self.pos = position

    def update(self, key, value):
        self.cache_k[:, :, self.pos] = key
        self.cache_v[:, :, self.pos] = value
        return self.cache_k, self.cache_v 

    def state(self):
        return self.cache_k, self.cache_v

    @classmethod
    def empty(cls, shape, device):
        k = torch.zeros(shape).cuda()
        v = torch.zeros(shape).cuda()
        return cls(k, v, 0)
        