"""Read-only: estimate progress of xiang's 4 running 100-epoch jobs from their logs (count completed epochs from the
table rows, elapsed time from process etime).  Never touches the jobs."""
import re, subprocess, glob

LOGS = "/works/xiang/MALWARE_100EP_DIFFGM/_logs"
for name in ["EssicaNet31S11__seed2029", "EssicaNet31S11__seed2030", "EssicaNet31S9__seed2029", "EssicaNet31S9__seed2030"]:
    f = f"{LOGS}/virusmnist__{name}.log"
    txt = open(f, "rb").read().decode("utf-8", "replace").replace("\r", "\n")
    lines = [l for l in txt.split("\n") if l.strip()]
    keep = [l for l in lines if not l.startswith("Training @") and not l.startswith("Validat") and "batchs/s" not in l]
    # one full 0..757 progress bar per epoch: count bars that reached 757/757
    done = len(re.findall(r"Training @[^\n]*\b757/757\b", txt))
    print("==", name, "train_bars_complete", done)
    for l in keep[-6:]:
        print("   ", l[:150])
