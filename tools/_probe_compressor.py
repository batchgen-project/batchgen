import torch.nn as nn

from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor

for overlap in (False, True):
    try:
        c = DeepSeekV4Compressor(
            4096, 512, 64, 128, 1e-6, overlap=overlap, rotate=False
        )
        print("overlap", overlap, "OK wkv.weight", tuple(c.wkv.weight.shape))
    except Exception as e:
        print("overlap", overlap, "FAIL", type(e).__name__, str(e))

probe = nn.Linear(4096, 512, bias=False)
print(
    "plain nn.Linear weight dims", probe.weight.dim(), tuple(probe.weight.shape)
)
