# Occlusion

Benchmark code and human data for [Understanding the Robustness of Vision Models and Humans to Occlusion-Based Corruptions](https://hdl.handle.net/1721.1/151342) (MIT, 2023).

The benchmark varies occlusion severity and spatial scale. It includes released masks and human responses, model evaluation and finetuning, inpainting baselines, and adversarial occlusion attacks.

## Generate benchmark images

Obtain ImageNet validation images separately, organized as `<val>/<synset>/*.JPEG`. With PyTorch and torchvision installed:

```bash
python -m pip install numpy Pillow pandas scipy matplotlib seaborn tqdm \
    PyYAML pycocotools pyfastnoisesimd scikit-image grad-cam adjustText scikit-learn
(cd src/pytorch-image-models && python -m pip install -e .)
python scripts/generate_benchmark.py --imagenet-dir /path/to/imagenet/val
```

This reconstructs **18,400 images from 400 source images across 46 conditions**, using the released masks and original 224 × 224 preprocessing. Images are written to `data/human_occ_exp/hexp1.1/`; existing images are skipped. Use `--cropped` if supplying the original `val_cropped` JPEGs. Model weights are unnecessary for image generation.

## Data and experiments

- `data/human_occ_exp/hexp1.1/` — masks, conditions, labels, and subject assignments.
- `results/hexp1.1/human/` — human responses for occluded, interpolation-inpainted, and diffusion-inpainted images.
- `scripts/inference.py`, `scripts/train.py` — machine predictions and occlusion finetuning.
- `scripts/mk_hexp.py` — fresh stimulus and subject-assignment generation; configure its paths and experiment settings before running.
- `scripts/*inpaint*.py`, `scripts/*attack*.py` — inpainting and adversarial experiments.
- `notebooks/` — model/human comparisons, attack analysis, and inpainting visualization.
- `human_experiment/` — response collection interface; `src/` contains bundled model implementations.

Run legacy experiments from `scripts/`, using the bundled timm and the required pretrained checkpoints. Configure data and cluster paths for your machine. For example, evaluate a pretrained ResNet on the released 50%-occluded, frequency-8 condition:

```bash
cd scripts
python inference.py --model resnet50 --pretrained --dataset imagenet \
    --data-dir ../data/human_occ_exp/hexp1.1/orig_cropped_images \
    --input-size 3 224 224 --skip-crop --occ-transform precomputed \
    --mask-path ../data/human_occ_exp/hexp1.1/simplex_noise_8_25088/masks.json \
    --gt-loc ../data/human_occ_exp/hexp1.1/image_labels.csv \
    --results-dir ../results/benchmark
```

The exporter writes `image_labels.csv` with one label per source image. Default inference uses the benchmark's 50 classes; add `--all-class-eval` for all 1,000 ImageNet classes.

Finetuning uses the full ImageNet training split; the historical recipe is 20 epochs, batch 128/GPU, SGD with momentum 0.9, LR 0.001/GPU, weight decay 2e-5, and LR decay ×0.9 each epoch. CLIP entries are ImageNet-finetuned image classifiers, not zero-shot text-prompt classifiers.
