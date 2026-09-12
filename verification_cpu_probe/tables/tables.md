# Measured results (seed 0)

## TABLE IV - PSNR / SSIM at x4

| Method | RS-Bench PSNR | RS-Bench SSIM |
|---|---|---|
| Bilinear | 32.0975 | 0.8170 |
| Bicubic | 32.0701 | 0.8118 |
| Lanczos | 31.9615 | 0.8064 |
| Bicubic + unsharp | 31.5671 | 0.7895 |
| Back-projection | 30.8594 | 0.7593 |

## TABLE V - perceptual quality on RS-Bench

| Method | LPIPS | MOS | MSE |
|---|---|---|---|
| Bilinear | - | - | 53.8167 |
| Bicubic | - | - | 52.8308 |
| Lanczos | - | - | 53.6700 |
| Bicubic + unsharp | - | - | 57.9990 |
| Back-projection | - | - | 69.3232 |

## TABLE VI - confusion matrix, Back-projection

| Actual \ Pred. | Wat | Veg | Bui | Bar | Cro | Tot | Rec % |
|---|---|---|---|---|---|---|---|
| Water | 488 | 12 | 0 | 0 | 0 | 500 | 97.60 |
| Vegetation | 55 | 194 | 0 | 0 | 251 | 500 | 38.80 |
| Built-up | 266 | 12 | 0 | 10 | 212 | 500 | 0.00 |
| Barren | 113 | 0 | 0 | 32 | 355 | 500 | 6.40 |
| Cropland | 87 | 60 | 0 | 10 | 343 | 500 | 68.60 |

## TABLE VII - per-class metrics

| Class | Support | Prec. | Recall | F1 | IoU |
|---|---|---|---|---|---|
| Water | 500 | 0.4836 | 0.9760 | 0.6468 | 0.4780 |
| Vegetation | 500 | 0.6978 | 0.3880 | 0.4987 | 0.3322 |
| Built-up | 500 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Barren | 500 | 0.6154 | 0.0640 | 0.1159 | 0.0615 |
| Cropland | 500 | 0.2954 | 0.6860 | 0.4130 | 0.2602 |
| Macro average | 2500 | 0.4185 | 0.4228 | 0.3349 | 0.2264 |

## TABLE VIII - downstream LULC classification by input condition

| Input condition | OA (%) | Macro F1 | mIoU | kappa |
|---|---|---|---|---|
| LR (x4 down) | 46.4800 | 0.3763 | 0.2915 | 0.3310 |
| Bilinear | 39.6000 | 0.2687 | 0.1868 | 0.2450 |
| Bicubic | 39.8800 | 0.2765 | 0.1912 | 0.2485 |
| Lanczos | 40.0800 | 0.2833 | 0.1946 | 0.2510 |
| Bicubic + unsharp | 39.5200 | 0.3002 | 0.1984 | 0.2440 |
| Back-projection | 42.2800 | 0.3349 | 0.2264 | 0.2785 |
| HR reference | 99.1200 | 0.9912 | 0.9826 | 0.9890 |

## TABLE TUG - task-utility gap (OA) with 95% bootstrap CI

| name | psnr | ssim | lpips | oa | tug | tug_ci_low | tug_ci_high | tur | rank_psnr | rank_tug |
|---|---|---|---|---|---|---|---|---|---|---|
| Bilinear | 32.0963 | 0.8145 | - | 0.3960 | 1.1307 | 1.1103 | 1.1527 | -0.1307 | 1.0000 | 4.0000 |
| Bicubic | 32.0925 | 0.8101 | - | 0.3988 | 1.1254 | 1.1038 | 1.1469 | -0.1254 | 2.0000 | 3.0000 |
| Lanczos | 31.9968 | 0.8052 | - | 0.4008 | 1.1216 | 1.1002 | 1.1435 | -0.1216 | 3.0000 | 2.0000 |
| Bicubic + unsharp | 31.6327 | 0.7897 | - | 0.3952 | 1.1322 | 1.1067 | 1.1577 | -0.1322 | 4.0000 | 5.0000 |
| Back-projection | 30.9649 | 0.7619 | - | 0.4228 | 1.0798 | 1.0561 | 1.1031 | -0.0798 | 5.0000 | 1.0000 |

Selected model by criterion: by_tug -> Back-projection, by_psnr -> Bilinear, by_ssim -> Bilinear
- Kendall tau(TUG, psnr) = -0.400 (p = 0.483); rank reversals: 7
- Kendall tau(TUG, ssim) = -0.400 (p = 0.483); rank reversals: 7

## Significance
```json
{
  "ttest_psnr": {
    "bicubic": {
      "n": 1500,
      "mean_diff": -1.2106669590498178,
      "std_diff": 1.1773264071828167,
      "t": -39.82661852819057,
      "p": 3.2203349575142554e-237,
      "cohen_dz": -1.0283188686362523
    },
    "lanczos": {
      "n": 1500,
      "mean_diff": -1.10208109590948,
      "std_diff": 0.9891142367042578,
      "t": -43.153172527873096,
      "p": 3.963981889779819e-265,
      "cohen_dz": -1.114210123576453
    }
  },
  "mcnemar": {
    "backprojection_vs_bicubic": {
      "a_right_b_wrong": 150,
      "a_wrong_b_right": 90,
      "chi2_cc": 14.504166666666666,
      "p_chi2": 0.0001398498746272035,
      "p_exact": 0.00012987667540779012,
      "p": 0.0001398498746272035
    }
  },
  "seeds": {}
}
```
