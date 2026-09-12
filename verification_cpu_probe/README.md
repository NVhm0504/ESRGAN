# CPU verification run (no PyTorch)

Produced in the build environment to prove the dataset generator and the NumPy evaluation stack work end to end on the full `paper` preset (12,000 + 2,500 patches):

    python scripts/make_dataset.py --preset paper --out datasets
    python scripts/make_classical_baselines.py --config configs/probe_demo.yaml --extra
    python scripts/evaluate_reconstruction.py  --config configs/probe_demo.yaml
    python scripts/evaluate_task_probe.py      --config configs/probe_demo.yaml
    python scripts/tug_report.py / significance.py / make_tables.py / make_figures.py --config configs/probe_demo.yaml

These numbers come from non-learned reconstructions (nearest, bilinear, bicubic, Lanczos, unsharp-masked bicubic, iterative back-projection) scored by a hand-crafted texture/colour descriptor with gradient boosting, trained on HR. They are NOT ESR-GAN or ResNet-50 results and must not be put in the paper. Those come from running `run_pipeline.py --config configs/paper.yaml` on a GPU.

What the run shows:
* The HR task set is separable (probe OA 99.1 %) and the Eq. (1) degradation removes the cues (probe OA 46.5 % on LR).
* Every interpolator raises PSNR by about 1 dB over LR yet lowers probe accuracy (TUG > 1). Back-projection has the lowest PSNR of the five yet the best TUG. PSNR and TUG disagree on 7 of 10 model pairs (Kendall τ = −0.40), which is the rank-reversal behaviour the TUG metric is designed to expose.
* The hand-crafted probe sends most smooth inputs to "cropland". A CNN is expected to degrade differently, so do not read class-level structure from these matrices.
