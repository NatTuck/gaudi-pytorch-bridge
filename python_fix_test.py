"""Verify the issue-23 defect-1 fix by monkeypatch: does accepting `metadata` make the
torch-level export work? If yes, the two-line patch is correct."""
import os, json, collections
import torch, habana_frameworks.torch  # noqa
from torch.profiler import ProfilerActivity, profile

@torch.compile(backend="hpu_backend", dynamic=False, fullgraph=True)
def f(a, b): return torch.relu(a @ b)

a = torch.randn(512, 1024, dtype=torch.bfloat16, device="hpu")
b = torch.randn(1024, 1024, dtype=torch.bfloat16, device="hpu")
for _ in range(3): f(a, b)
torch.hpu.synchronize()

p = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.HPU])
p.start()
for _ in range(5): f(a, b)
torch.hpu.synchronize()
p.stop()

# The shim's method takes (self, path). Torch calls (self, path, metadata).
cls = type(p.profiler)
orig = cls.export_chrome_trace
print("class:", cls.__name__)
print("original signature takes:", orig.__code__.co_varnames[:orig.__code__.co_argcount])

# apply the candidate fix
cls.export_chrome_trace = lambda self, path, *args, **kw: orig(self, path)

out = "/tmp/fix-test/trace.json"
os.makedirs("/tmp/fix-test", exist_ok=True)
try:
    p.export_chrome_trace(out)          # this is the call that raised TypeError
    ev = json.load(open(out))["traceEvents"]
    cats = collections.Counter(e.get("cat") for e in ev)
    reg = sum(1 for e in ev if "Torch-Compiled Region" in str(e.get("name", "")))
    dev = sum(1 for e in ev if e.get("cat") in ("kernel", "hpu_op"))
    print(f"EXPORT OK: {len(ev)} events, region spans {reg}, device events {dev}")
    print("FIX VERIFIED" if dev and reg else "FIX INSUFFICIENT (no device events or no regions)")
except TypeError as e:
    print("STILL FAILS:", e)
