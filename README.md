<div align="center">

<img src="docs/assets/logo.png" alt="Relay-OPD" height="90">

# Pass the Baton: Trajectory-Relayed On-Policy Distillation

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zju-real.github.io/Relay-OPD/)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b)](#)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](#)

**Haolei Xu<sup>1,2*</sup> · Yongliang Shen<sup>1†</sup> · Weiming Lu<sup>1†</sup>**

<sup>1</sup>Zhejiang University &nbsp; <sup>2</sup>Alibaba Group

</div>

---

**Relay-OPD** fixes *prefix failure* in on-policy distillation. A label-free **handoff trigger** — the teacher prefers a reflection token that is absent from the student's top-K — locates failed prefixes online during student generation. The teacher then briefly takes over for a short **teacher leg** (executed speculatively: the student keeps drafting, the teacher verifies with one forward pass per leg), hands the trajectory back, and a limited **relay budget (M, L)** keeps intervention early and local. The student is optimized on the relayed trajectory, including the relay tokens themselves.

With a Qwen3-4B-Instruct-2507 teacher and Qwen3-0.6B/1.7B-Non-Thinking students on eight mathematical reasoning benchmarks, Relay-OPD achieves the best or second-best result on every benchmark — **+5.73%** over standard OPD and **+1.49%** over the strongest baseline FastOPD on average at 1.7B — while cutting average training trajectory length by **more than 50%**.

<div align="center">
<img src="docs/assets/teaser.png" alt="Relay-OPD teaser" width="95%">
</div>

## News

- **2026-07** — Preprint and project page released. Code is being cleaned up and will be available soon.

## Links

- 📄 Paper: arXiv link coming soon
- 🌐 Project page: <https://zju-real.github.io/Relay-OPD/>

## Citation

```bibtex
@misc{xu2026relayopd,
  title={Pass the Baton: Trajectory-Relayed On-Policy Distillation},
  author={Haolei Xu and Yongliang Shen and Weiming Lu},
  year={2026},
  url={https://github.com/zju-real/Relay-OPD},
}
```
