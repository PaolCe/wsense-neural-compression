# WSense Neural Compression Challenge

Hybrid compression pipeline for high-resolution underwater images: AVIF at very aggressive settings, followed by a small U-Net that restores the image on the decoder side.

**Result:** leaderboard score 1.06 in the challenge organised by WSENSE with the MSc in Data Science at Sapienza University of Rome (September 2026).

**Team Underwater Overfitters:** Andrea Scalise, Mattia Mungo, Francesco La Piana, Paolo Cencia.

## The problem

The challenge had three tasks on underwater images:

| Task | Goal | Reference values |
|---|---|---|
| 1 | Compress single images at one fixed operating point | 34 dB PSNR, 0.975 MS-SSIM, CR 120 |
| 2 | Fit every image into a hard byte budget | 30 dB PSNR, 0.930 MS-SSIM, CR 400 |
| 3 | Compress groups of images from the same scene | 34 dB PSNR, 0.975 MS-SSIM, CR 150 |

## Key idea

Quality enters the score linearly, while compression enters through a logarithm with a large weight. Halving the payload is worth about 6.6 dB of PSNR, and in practice it costs only 1 to 3 dB. The best operating point is therefore at far higher compression than intuition suggests.

We first trained an end-to-end autoencoder with a hyperprior, which reached about 0.85. Plain AVIF with no network scored higher in our local measurements, so we moved to a hybrid approach.

## Pipeline

**Encoder**
1. Downscale the image: to 0.25 of its original size on Tasks 1 and 3, and to 0.60 on Task 2.
2. Compress it with AVIF. Tasks 1 and 3 use quality 20. Task 2 uses a binary search for the highest quality that fits within 5,500 bytes.
3. Add a 9-byte header.

In the final submission, payloads averaged about 6.1 KB on Task 1 (1024×2048), 5.3 KB on Task 2 and 1.3 KB on Task 3 (1080×1920).

**Decoder**
1. Decode AVIF and upscale to the original size.
2. Run a U-Net restorer on overlapping 512×512 tiles.

The restorer has 1.26M parameters. It predicts a residual, uses FiLM conditioning on quality and scale, and is trained with `50·MSE + (1 − MS-SSIM)`. The network is part of the shared model, so each image pays only for its AVIF bytes.

## Lesson learned

PSNR once stayed near 24 dB while compression varied 76-fold. The tile grid never reached the image border, which left 7.7% of the pixels gray on 1024×2048 images. After the fix, Task 1 gained about 5 dB and Task 3 about 15 dB, with the same weights and the same bytes.

## Repository contents

| File | Purpose |
|---|---|
| `hybrid_codec.py` | Bitstream format, restorer network, the three codecs and the operating-point sweeps |
| `train_hybrid.ipynb` | Trains the restorer and measures the operating point |
| `submission_hybrid.ipynb` | Builds `submission.csv` |
| `wsense_operating_point.json` | Selected operating point |

## Running it

The code was written for Kaggle notebooks with a GPU. It needs the competition data and `challenge_common.py`, the organisers' evaluation harness, which are not included in this repository. Dependencies are Python, PyTorch, NumPy, SciPy, pandas and Pillow with AVIF support.
