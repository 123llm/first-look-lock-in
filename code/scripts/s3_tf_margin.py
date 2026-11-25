"""S3 diagnostic: for trajectories where the teacher-forced letter != rollout letter, print the letter softmax
(eval mode, no grad) and compare the re-tokenised prompt with the token ids produced during generation."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from tpa.data.vstar import load_vstar, load_image01
from tpa.models.qwen_agent import QwenAgent, Trajectory, LETTERS

root = "."
ids = sys.argv[2].split(",")
items = {s.image_id: (s, p) for s, p, _ in load_vstar(f"{root}/datasets/craigwu__vstar_bench")}
trajs = {json.loads(l)["id"]: json.loads(l) for l in open(f"{root}/results/S3/model_check_deepeyes_traj.jsonl")}
agent = QwenAgent(sys.argv[1])
for i in ids:
    s, p = items[i]
    t = trajs[i]
    tr = Trajectory(segments=t["segments"], boxes=[tuple(b) if b else None for b in t["boxes"]])
    x = load_image01(p).to("cuda")
    with torch.no_grad():
        lg = agent.final_answer_logits(x, s, tr)
    pr = torch.softmax(lg, 0).tolist()
    print(i, "rollout", t["final"], "tf", LETTERS[int(lg.argmax())], "probs", [round(v, 4) for v in pr], flush=True)
