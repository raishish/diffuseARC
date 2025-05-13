# diffuseARC
Discrete Diffusion for ARC-AGI

This repository includes three approaches for using masked discrete diffusion language models (MDLMs) for abstraction and reasoning, particularly the ARC-AGI challenge:

1. Pre-trained models

    a. Pre-trained masked discrete diffusion language model (LLaDA)

    b. Pre-trained enterprise grade masked diffusion language model trained on code (Mercury Coder)

2. Training masked discrete language models from scratch for image inpainting on ARC-AGI tasks

## Directory Structure
1. [llada](llada): Contains the code for experiments with LLaDA
2. [mercury-coder](mercury-coder): Contains the code for experiments with InceptionLabs Mercury Coder Small
3. [inpaintARC](inpaintARC): Contains the code for inpainting experiments

README.md for experiments in respective directories.

## Citation & Licence

License: MIT.

If you use or reference this code, please cite:

```
@article{rai2025discrete-diffusion-for-arc-agi,
  title={Discrete Diffusion for Abstraction and Reasoning},
  author={Rai, Ashish and Zhong, Sichen and Jain, Shraddha},
  year={2025},
  month={May}
}
```