import sys, os
os.environ["PYTHONNOUSERSITE"] = "1"
sys.path.insert(0, "D:/claudeAgent/RAG")

from PIL import Image
from transformers import AutoProcessor

proc = AutoProcessor.from_pretrained(
    "D:/claudeAgent/RAG/florence-2-large",
    trust_remote_code=True,
)

img   = Image.new("RGB", (224, 224), color=(128, 128, 128))
task  = "<MORE_DETAILED_CAPTION>"

inputs = proc(text=task, images=img, return_tensors="pt")
print("Keys:", list(inputs.keys()))
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"  {k}: tensor shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
