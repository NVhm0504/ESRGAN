# ESR-GAN + Task-Utility Gap (TUG) — dataset and code

Companion code for *"Task-Utility Gap (TUG): A Downstream-Anchored Metric for Selecting Super-Resolution Models"* (N. Varshney, H. Mathur, RNTU Bhopal). It covers every experiment in the manuscript: the degradation model (Eq. 1), the RRDB-DA generator with channel-spatial attention (Eqs. 3–5), the relativistic average discriminator (Eqs. 6–8), the five-term objective (Eqs. 9–13), the frozen ResNet-50 task protocol (Section III-E, Algorithm 1), all eight comparison methods, the A–D ablation, complexity and significance tests. It also adds the TUG metric named in the title.

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
python selftest.py        # RUN THIS FIRST: offline, ~2 min on GPU / ~10 min on CPU
```

`selftest.py` checks every model, loss and script end to end on a tiny generated dataset. If it prints `N/N checks passed`, the full pipeline will run.

## 2. Dataset

### 2a. Generated dataset (default)
```bash
python scripts/make_dataset.py --preset paper --out datasets   # ~10 min on 2 CPU cores, ~1.3 GB
python scripts/make_dataset.py --preset mini  --out datasets_mini
```

| Set | Count | Contents |
|---|---|---|
| RS-12K train | 9,500 | 192×192 HR patches, 1,900 per class, 190 regions |
| RS-12K val | 1,000 | 20 held-out regions, fixed Eq. (1) LR |
| RS-12K test (= RS-Bench) | 1,500 | 30 held-out regions, fixed Eq. (1) LR |
| LULC task set | 2,500 | 500 per class, drawn from the test regions |
| Showcase | 10 | 384×384 mixed scenes for the qualitative figure |

Each patch has an HR image, a per-pixel land-cover map (`LABEL/`) and a manifest row. The row stores region, scene type (Harbour, Urban-Grid, Farmland, River-Delta, Quarry), dominant-class purity and the exact degradation parameters. The simulator (`esrgan_tug/data/simulator.py`) renders water, vegetation (tree crowns with cast shadows, meadow), built-up (rotated road grids, roofs, shadows), barren (soil, gullies, tracks, quarry benches) and cropland (parcels with 4.5–11 px crop rows and field margins). The design builds in the paper's central assumption:

* vegetation and cropland share a green palette and differ mainly in straight parcel edges and row texture;
* barren and built-up share a grey-brown palette and differ mainly in rectilinear footprints and shadows;
* about 20–25 % of patches per class are realistic confusers: hedgerows in meadows, large weakly-rowed parcels, haul roads and stockpiles on bare land, and sparse peri-urban lots.

These cues sit above the frequencies that ×4 decimation preserves. Region styles (palettes, sun angle, haze, road orientation, crop calendar) never cross splits, so the splits have no geographic overlap. Generation is deterministic for a given `--seed` (the delivered sample used seed 2026 with NumPy 2.4 and OpenCV 4.13; a different OpenCV JPEG codec can shift LR pixels slightly, while HR and labels are unaffected). `datasets_mini/` (a small class-balanced sample from the same generator: 225 train / 25 val / 35 test patches, 100 LULC patches, 5 showcase scenes) is included for the smoke run.

Training LR is **not** stored: Eq. (1) is re-sampled on the fly for every patch (anisotropic Gaussian σ∈[0.6,2.4], ×4 decimation, Gaussian noise σ∈[0,12]/255, JPEG q∈[45,95]).

### 2b. Real imagery (optional)
`scripts/prepare_real_data.py` writes the same layout from AID / NWPU-RESISC45 / UC Merced (5-class folder mapping, split by source image), DIV2K sub-images, and Set5/Set14/BSD100/Urban100 with MATLAB-bicubic ×4 LR. Download those datasets yourself and check their licences. Any test set whose folder is missing is skipped automatically.

**Reporting note:** the manuscript says RS-12K was "assembled from publicly available aerial orthophotography". If the generated dataset is used, the Datasets section must say so (a procedurally simulated 0.5 m corpus). If 2b is used, name the source datasets instead.

## 3. Run the experiments

```bash
python scripts/run_pipeline.py --config configs/smoke.yaml    # minutes: whole pipeline on datasets_mini
python scripts/run_pipeline.py --config configs/quick.yaml    # 2 % of the iterations, same day on one GPU
python scripts/run_pipeline.py --config configs/paper.yaml --seeds 0 1 2 --lpips   # full study
```
The driver skips finished steps, so it can be relaunched after an interruption (training resumes from checkpoints). Individual steps:

| Paper item | Script |
|---|---|
| Alg. 1, Table III (train one model) | `train_sr.py --model esrgan_da` (also `srcnn vdsr srresnet srgan esrgan realesrgan ablation_A/B/C`) |
| Frozen ResNet-50 (III-E) | `train_classifier.py` |
| SR outputs | `infer.py --model …`; LR / bicubic / HR conditions: `make_classical_baselines.py` |
| Tables IV, V; Figs. 3–5 | `evaluate_reconstruction.py --lpips` |
| Tables VI–VIII; Figs. 8–10 | `evaluate_task.py` |
| TUG table, rank reversals, Fig. 13 | `tug_report.py` |
| Section V-H (t-test, McNemar, seeds) | `significance.py --seeds 0 1 2` |
| Table X, Fig. 12 | `complexity_report.py` |
| Table IX, Fig. 11 | produced from the ablation runs by `make_tables.py` / `make_figures.py` |
| All tables / figures | `make_tables.py` → `runs/seed0/tables/tables.md`; `make_figures.py` → `runs/seed0/figures/` |

Loss-weight grid search: `train_sr.py --model esrgan_da --set gan.weights.freq=0.1 gan.weights.edge=0.05`.

**Compute budget (one RTX A5000).** The proposed model takes about 52 h (500k pre-train + 400k adversarial). ESRGAN and Real-ESRGAN (23 blocks) take roughly twice that. All ten trained models plus three seeds of the proposed model come to about 2–3 weeks. Ablations B and C reuse the proposed model's pre-training.

## 4. TUG metric

For a frozen task model evaluated on the LR input, a candidate reconstruction and the HR reference (task score T, default overall accuracy):

    TUG = (T_HR − T_SR) / (T_HR − T_LR)        lower is better;  TUR = 1 − TUG

TUG = 0 means the reconstruction is as useful as HR, 1 means no better than the LR input, and > 1 means the reconstruction harms the task. `tug_report.py` reports TUG for OA, macro-F1, κ and mIoU, per-class TUG, and a class-stratified paired bootstrap 95 % CI. It also gives Kendall τ / Spearman ρ between the TUG ranking and the PSNR/SSIM/LPIPS rankings, every rank-reversed pair, and the model each criterion would select. The paper's "recovers 88 % of the accuracy the degradation destroys" is TUR = (92.68 − 74.16)/(95.24 − 74.16) = 0.879.

## 5. Where the manuscript and the specified architecture disagree

The code builds exactly what Sections III–IV specify, so the measured values will not match these statements in the draft:

| Manuscript statement | What the specification actually gives |
|---|---|
| Generator has 12.34 M parameters (Table X), 26 % fewer than ESRGAN | 8 RRDB-DA blocks with nf = 64, gc = 32 and a PixelShuffle upsampler give **6.10 M** (63 % fewer). 12.34 M would need about 16.7 blocks. ESRGAN's 16.70 M is correct. |
| Attention gate adds ≈ 0.31 M parameters | CBAM with r = 16 at 64 channels adds 679 parameters per block, **5.4 K** for 8 blocks |
| FLOPs 214.6 G (proposed) and 292.4 G (ESRGAN) at a 256×256 LR tile | Multiply-accumulates at 256×256 LR are about **430 G** and **1,175 G** |
| LULC set of 2,500 patches "drawn from" RS-Bench | RS-Bench has only 1,500 patches. Here the task set is drawn from the same held-out test regions |
| Table IX, config A has 88.04 % OA, identical to ESRGAN in Table VIII | A is an 8-block RRDB model, not ESRGAN (29.21 vs 28.21 dB); identical OA is unlikely |

All metric formulas (Eqs. 14–19) are implemented as written. `tests/test_numpy_core.py` reproduces Table VII exactly from the Table VI confusion matrix, so those two tables are internally consistent.

## 6. Layout
```
esrgan_tug/data/        simulator.py, degradation.py (Eq. 1, MATLAB imresize, Real-ESRGAN high-order), datasets.py
esrgan_tug/models/      rrdbnet.py (RRDB-DA), baselines.py, discriminators.py, registry + ResNet-50 wrapper
esrgan_tug/losses.py    L_pix, L_per (VGG19 conv5_4 pre-act), RaGAN, L_freq (FFT magnitude), L_edge (Sobel + TV)
esrgan_tug/metrics/     image_quality.py, classification.py, tug.py, stats.py   (NumPy - no GPU)
scripts/                all runnable steps (see table above)
configs/                paper.yaml (full), quick.yaml, smoke.yaml, probe_demo.yaml (CPU-only check)
tests/                  NumPy unit tests:  python -m unittest discover tests
selftest.py             offline end-to-end check of everything incl. PyTorch
```
