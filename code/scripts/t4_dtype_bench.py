"""T4: matmul throughput bf16 vs fp16 vs fp32, and NF4 linear forward with bf16 vs fp16 compute dtype.
Engineering check for the T4 server (sm_75 has fp16 tensor cores but no native bf16)."""
import time, torch

def bench(f, n=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        f()
    torch.cuda.synchronize(); return (time.time() - t) / n

for dt in (torch.float16, torch.bfloat16, torch.float32):
    a = torch.randn(4096, 4096, device="cuda", dtype=dt); b = torch.randn(4096, 4096, device="cuda", dtype=dt)
    s = bench(lambda: a @ b)
    print(f"matmul 4096^3 {str(dt):15s} {s*1e3:7.1f} ms  {2*4096**3/s/1e12:5.1f} TFLOPS", flush=True)

import bitsandbytes as bnb
for dt in (torch.float16, torch.bfloat16):
    lin = bnb.nn.Linear4bit(3584, 18944, compute_dtype=dt, quant_type="nf4").cuda()
    for bs in (1, 512):
        x = torch.randn(bs, 3584, device="cuda", dtype=dt)
        s = bench(lambda: lin(x))
        print(f"nf4 linear 3584x18944 bs={bs:4d} compute {str(dt):15s} {s*1e3:7.2f} ms", flush=True)
