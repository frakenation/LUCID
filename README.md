<p align="center">
  <img src="figs/LUCID_logo.png" width="320" alt="LUCID logo">
</p>

<h1 align="center">
  LUCID: Learning Unified Control for Image Deflaring <br>
  and Exposure Mastery in Nighttime Photography
</h1>

<p align="center">
  Tingyu Yang<sup>1</sup>,
  <a href="https://cyuan328.github.io/">Yuan Cheng</a><sup>1</sup>,
  <a href="https://xiaoyunyuan.net/">Xiaoyun Yuan</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Shanghai Jiao Tong University, Shanghai, China
</p>

<p align="center">
  Accepted to <b>ACM SIGGRAPH 2026</b>
</p>

<p align="center">
  <a href="https://xiaoyunyuan.net/index.html?project=lucid">
    <img src="https://img.shields.io/badge/Project-Page-blue">
  </a>
  <a href="https://arxiv.org/abs/2606.06901">
    <img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv">
  </a>
  <a href="https://arxiv.org/pdf/2606.06901">
    <img src="https://img.shields.io/badge/PDF-arXiv-b31b1b">
  </a>
  <a href="https://www.youtube.com/watch?v=AGPLSiZcK_I">
    <img src="https://img.shields.io/badge/Video-YouTube-red?logo=youtube">
  </a>
  <a href="https://player.bilibili.com/player.html?bvid=BV1DvEK6EEzu&page=1&autoplay=0">
    <img src="https://img.shields.io/badge/Video-Bilibili-00a1d6">
  </a>
</p>

---

> **Abstract:** Photography is the art of painting with light, yet nighttime scenes are shaped by competing degradations: intense flares obscure scene structure, while photon-limited regions collapse into noise. Conventional approaches address these factors in isolation, overlooking the fact that these degradations are fundamentally entangled. We introduce **LUCID**, a unified framework that reframes nighttime restoration as a continuous and controllable process rather than a fixed correction. LUCID restores challenging nighttime images with flexible control over exposure, light sources, flare, and ghosting artifacts, while also supporting high dynamic range (HDR) reconstruction.

<p align="center">
  <img src="figs/teaser.png" alt="LUCID teaser" width="95%">
</p>

## Model Capabilities

LUCID is designed for nighttime image restoration where underexposure, flare, ghosting, and light-source artifacts are coupled. Instead of treating low-light enhancement and flare removal as separate problems, LUCID learns a unified controllable restoration process.

- **Deflare + Low-Light Enhancement:** restore dark, flare-contaminated nighttime scenes in one model.
- **Continuous Exposure Control:** synthesize smoothly varying exposure states from a single input.
- **Light-Source Editing:** preserve visible illuminants or suppress them with associated flare.
- **Single-Image HDR Reconstruction:** use controllable exposure synthesis to recover HDR-style outputs.
- **Creative Downstream Use:** clean flare and exposure artifacts before editing or generation workflows.

## Continuous Exposure Control

LUCID enables continuous output exposure modulation through CFG-scale control. The same input can be restored into different illumination states while preserving scene structure.

<div align="center">
<table>
  <tr>
    <td align="center"><img src="figs/control_06773/input.jpg" width="180"><br><b>Input</b></td>
    <td align="center"><img src="figs/control_06773/cfg_050.png" width="180"><br><b>β = 0.50</b></td>
    <td align="center"><img src="figs/control_06773/cfg_105.png" width="180"><br><b>β = 1.05</b></td>
    <td align="center"><img src="figs/control_06773/cfg_150.png" width="180"><br><b>β = 1.50</b></td>
  </tr>
  <tr>
    <td align="center"><img src="figs/control_03600/input.jpg" width="180"><br><b>Input</b></td>
    <td align="center"><img src="figs/control_03600/cfg_050.png" width="180"><br><b>β = 0.50</b></td>
    <td align="center"><img src="figs/control_03600/cfg_105.png" width="180"><br><b>β = 1.05</b></td>
    <td align="center"><img src="figs/control_03600/cfg_150.png" width="180"><br><b>β = 1.50</b></td>
  </tr>
</table>
</div>

## Light-Source Control

LUCID can preserve visible light sources or suppress them together with associated flare and ghosting artifacts.

<p align="center">
  <img src="figs/light_input.png" width="30%" alt="Input with light source">
  <img src="figs/light_preserve.png" width="30%" alt="Preserve light source">
  <img src="figs/light_remove.png" width="30%" alt="Suppress light source">
</p>

<p align="center">
  <b>Input</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Preserve Source</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <b>Suppress Source</b>
</p>

## Representative Results

### Holistic Nighttime Restoration

LUCID restores visibility and structure across diverse nighttime scenes from ExDark, achieving cleaner restoration than low-light enhancement baselines.

<div align="center">
<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>ZeroDCE</b></td>
    <td align="center"><b>Retinexformer</b></td>
    <td align="center"><b>Reti-Diff</b></td>
    <td align="center"><b>DarkIR</b></td>
    <td align="center"><b>LUCID</b></td>
  </tr>
  <tr>
    <td><img src="figs/results_exdark_04400/input.png" width="130"></td>
    <td><img src="figs/results_exdark_04400/zerodce.png" width="130"></td>
    <td><img src="figs/results_exdark_04400/retinexformer.png" width="130"></td>
    <td><img src="figs/results_exdark_04400/reti_diff.png" width="130"></td>
    <td><img src="figs/results_exdark_04400/darkir.png" width="130"></td>
    <td><img src="figs/results_exdark_04400/lucid.png" width="130"></td>
  </tr>
  <tr>
    <td><img src="figs/results_exdark_03035/input.png" width="130"></td>
    <td><img src="figs/results_exdark_03035/zerodce.png" width="130"></td>
    <td><img src="figs/results_exdark_03035/retinexformer.png" width="130"></td>
    <td><img src="figs/results_exdark_03035/reti_diff.png" width="130"></td>
    <td><img src="figs/results_exdark_03035/darkir.png" width="130"></td>
    <td><img src="figs/results_exdark_03035/lucid.png" width="130"></td>
  </tr>
</table>
</div>

### Flare Mitigation

LUCID removes scattering artifacts while maintaining plausible light-source appearance, balancing flare suppression and scene fidelity.

<div align="center">
<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>GT</b></td>
    <td align="center"><b>Flare7K</b></td>
    <td align="center"><b>MFDNet</b></td>
    <td align="center"><b>Zhou et al.</b></td>
    <td align="center"><b>LUCID</b></td>
  </tr>
  <tr>
    <td><img src="figs/results_flare_000023/input.png" width="130"></td>
    <td><img src="figs/results_flare_000023/gt.png" width="130"></td>
    <td><img src="figs/results_flare_000023/flare7k.png" width="130"></td>
    <td><img src="figs/results_flare_000023/mfdnet.png" width="130"></td>
    <td><img src="figs/results_flare_000023/zhou.png" width="130"></td>
    <td><img src="figs/results_flare_000023/lucid.png" width="130"></td>
  </tr>
  <tr>
    <td><img src="figs/results_flare_000067/input.png" width="130"></td>
    <td><img src="figs/results_flare_000067/gt.png" width="130"></td>
    <td><img src="figs/results_flare_000067/flare7k.png" width="130"></td>
    <td><img src="figs/results_flare_000067/mfdnet.png" width="130"></td>
    <td><img src="figs/results_flare_000067/zhou.png" width="130"></td>
    <td><img src="figs/results_flare_000067/lucid.png" width="130"></td>
  </tr>
</table>
</div>

### Single-Image HDR Reconstruction

LUCID extends naturally to HDR reconstruction by synthesizing controllable pseudo-exposure sequences from one input.

<div align="center">
<table>
  <tr>
    <td align="center"><b>Input</b></td>
    <td align="center"><b>IntrinsicHDR</b></td>
    <td align="center"><b>LEDiff</b></td>
    <td align="center"><b>GasLight</b></td>
    <td align="center"><b>LUCID</b></td>
  </tr>
  <tr>
    <td><img src="figs/results_hdr_006/input.png" width="150"></td>
    <td><img src="figs/results_hdr_006/intrinsichdr.png" width="150"></td>
    <td><img src="figs/results_hdr_006/lediff.png" width="150"></td>
    <td><img src="figs/results_hdr_006/gaslight.png" width="150"></td>
    <td><img src="figs/results_hdr_006/lucid.png" width="150"></td>
  </tr>
  <tr>
    <td><img src="figs/results_hdr_079/input.png" width="150"></td>
    <td><img src="figs/results_hdr_079/intrinsichdr.png" width="150"></td>
    <td><img src="figs/results_hdr_079/lediff.png" width="150"></td>
    <td><img src="figs/results_hdr_079/gaslight.png" width="150"></td>
    <td><img src="figs/results_hdr_079/lucid.png" width="150"></td>
  </tr>
</table>
</div>

### Downstream Creative Applications

LUCID can be used before or after downstream editing/generation by removing distracting flare and recovering controllable nighttime illumination.

<p align="center">
  <img src="figs/creative.png" width="90%" alt="Creative workflow results">
</p>

---

## Contents

- [Installation](#installation)
- [Pretrained Weights](#pretrained-weights)
- [Inference](#inference)
- [Training](#training)
- [Citation](#citation)

## Installation

```bash
cd LUCID
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA environment if it is not already available.

## Pretrained Weights

| Component | Description | Download Link |
| --- | --- | --- |
| SD-Turbo | Pretrained one-step diffusion backbone | [Hugging Face](https://huggingface.co/stabilityai/sd-turbo) |
| Flare Disentanglement | Required for public LUCID inference | To be released |
| LUCID Restoration | Main restoration checkpoint | To be released |

Download SD-Turbo with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/stabilityai/sd-turbo /path/to/sd-turbo
```

Update `--pretrained_model_name_or_path` in the scripts with the downloaded checkpoint path.

## Inference

Single-scale restoration:

```bash
bash scripts/infer_lucid.sh
```

Multi-scale exposure control:

```bash
bash scripts/infer_lucid_cfg_index.sh
```

Save disentanglement components and an output grid:

```bash
bash scripts/infer_lucid_components.sh
```

Edit the explicit command-line arguments inside each script before running it.

## Training

Train flare disentanglement:

```bash
bash scripts/train_decomp.sh
```

Train LUCID restoration:

```bash
bash scripts/train_lucid.sh
```

The training script enables the mixing-state UNet through the explicit `--ms_unet` flag.

## Citation

If you find this project useful, please cite:

```bibtex
@inproceedings{yang2026lucid,
  title     = {LUCID: Learning Unified Control for Image Deflaring and Exposure Mastery in Nighttime Photography},
  author    = {Yang, Tingyu and Cheng, Yuan and Yuan, Xiaoyun},
  booktitle = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers '26)},
  year      = {2026},
  location  = {Los Angeles, CA, USA},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3799902.3811133},
  isbn      = {979-8-4007-2554-8}
}
```
