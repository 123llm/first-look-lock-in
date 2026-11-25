"""S3 diagnostic: compare the token ids seen by the model at the final <answer> step in (a) the rollout and
(b) the teacher-forced re-encoding of the decoded text."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from tpa.data.vstar import load_vstar, load_image01
from tpa.models.qwen_agent import QwenAgent, Trajectory

root = "."
items = {s.image_id: (s, p) for s, p, _ in load_vstar(f"{root}/datasets/craigwu__vstar_bench")}
agent = QwenAgent(sys.argv[1])
tok = agent.tok
for qid in sys.argv[2].split(","):
    s, p = items[qid]
    x = load_image01(p).to("cuda")
    # replicate rollout but keep ids
    traj = Trajectory()
    for turn in range(6):
        msgs, _ = agent._messages(s, traj, None)
        enc = agent._encode(msgs, agent._views(x, traj, None), add_generation_prompt=True)
        with torch.no_grad():
            out = agent.model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                       stop_strings=["</tool_call>", "</answer>"], tokenizer=tok,
                                       pad_token_id=tok.pad_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        seg = tok.decode(new, skip_special_tokens=True)
        from tpa.models.qwen_agent import parse_box
        box = parse_box(seg)
        if box is not None:
            from tpa.processors.qwen_diff import smart_resize
            h, w = x.shape[-2:]
            rh, rw = smart_resize(h, w, agent.spec.factor, agent.spec.min_pixels, agent.spec.max_pixels)
            box = (int(box[0] * w / rw), int(box[1] * h / rh), int(box[2] * w / rw), int(box[3] * h / rh))
        traj.segments.append(seg); traj.boxes.append(box)
        if box is None:
            break
    gen_ids = enc["input_ids"][0].tolist() + new.tolist()
    # teacher-forced encoding (same as final_answer_logits)
    last = traj.segments[-1]
    cut = last.split("<answer>")[0] + "<answer>"
    t2 = Trajectory(segments=traj.segments[:-1], boxes=traj.boxes[:-1])
    msgs, _ = agent._messages(s, t2, None)
    enc2 = agent._encode(msgs, agent._views(x, t2, None), add_generation_prompt=True, assistant_prefix=cut)
    tf_ids = enc2["input_ids"][0].tolist()
    n = len(tf_ids)
    print(qid, "len tf", n, "len gen", len(gen_ids), "prefix equal:", gen_ids[:n] == tf_ids)
    for k in range(min(n, len(gen_ids))):
        if gen_ids[k] != tf_ids[k]:
            print("  first diff at", k, "gen:", repr(tok.decode(gen_ids[k:k + 12])), "tf:", repr(tok.decode(tf_ids[k:k + 12])))
            print("  gen ids", gen_ids[k:k + 8], "tf ids", tf_ids[k:k + 8])
            break
    print("  next gen token after tf prefix:", repr(tok.decode(gen_ids[n:n + 3])))
    print("  rollout segments:", [g[-80:] for g in traj.segments])
