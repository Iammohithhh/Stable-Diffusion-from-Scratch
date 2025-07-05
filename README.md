# Stable Diffusion from Scratch

This repository contains a complete, modular PyTorch implementation of the Stable Diffusion pipeline, built from scratch as part of the Summer of Science 2025 program. The project reconstructs the core components of latent diffusion models—Variational Autoencoders (VAE), UNet, CLIP, and a denoising scheduler—enabling both text-to-image and image-to-image generation.

---

## Features

- Full implementation of Stable Diffusion architecture using PyTorch.
- Modular design: each component (CLIP, VAE, UNet, scheduler) is implemented in isolation and integrated via a custom pipeline.
- End-to-end generation capabilities from text and image prompts.
- Support for loading and integrating pretrained weights (Hugging Face compatible).


---

## Project Structure
Stable Diffusion from Scratch/
├── attention.py # Attention layers used in UNet
├── clip.py # CLIP text encoder
├── ddpm.py # Denoising Diffusion Probabilistic Model scheduler
├── decoder.py # VAE decoder
├── encoder.py # VAE encoder
├── diffusion.py # Diffusion sampling and reconstruction
├── model_loader.py # Pretrained weight loader
├── model_converter.py # Weight conversion logic
├── pipeline.py # End-to-end image generation pipeline
├── demo.ipynb # Notebook to test and demonstrate generation
├── data/ # Input text, captions, prompts
├── images/ # Output/generated images
├── req.txt # List of Python dependencies


---

## Setup Instructions

### 1. Clone the Repository

  ```bash
  git clone https://github.com/Iammohithhh/Stable-Diffusion-from-Scratch.git
  cd Stable-Diffusion-from-Scratch
 ```

### 2. Install the Dependencies

```bash
pip install -r req.txt
```
### 3.Run the Pipeline
```bash
python pipeline.py --prompt "A serene landscape with mountains during sunset"
```

## References
- Implementation guided by [Umar Jamil’s “Building Stable Diffusion from Scratch” series](https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=16548s)

Result:
![image](https://github.com/user-attachments/assets/b5a99698-4158-4336-8f6a-b06d2d366a90)


