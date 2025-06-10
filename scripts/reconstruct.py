import pathlib, shutil, torch, torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL

if torch.cuda.is_available():
    device, dtype = torch.device("cuda"), torch.float16
elif torch.backends.mps.is_available():
    device, dtype = torch.device("mps"), torch.float32
else:
    device, dtype = torch.device("cpu"), torch.float32
print(f"Using {device}, dtype {dtype}")

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float32).to(device).eval()

if device.type == "mps":
    vae.enable_tiling()

# Transforms
to_tensor = T.Compose([T.Resize(512, antialias=True), T.ToTensor()])
to_pil = T.ToPILImage()

def process_split(split_root: pathlib.Path):
    """
    Processes a dataset split by generating fake reconstructions of real images using a VAE.
    
    For each class directory inside `split_root`:
    - Creates two subdirectories: `0_real` for real images and `1_fake` for reconstructed images.
    - Moves any loose images into the `0_real` directory.
    - For each image in `0_real`, encodes it using the VAE and saves the reconstructed image in `1_fake`.
    """
    for cls_dir in [d for d in split_root.iterdir() if d.is_dir()]:
        # Create directories for real and fake images in each class directory
        real_dir = cls_dir / "0_real" 
        fake_dir = cls_dir / "1_fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        # Move any loose images inside the class folder into 0_real
        for p in cls_dir.iterdir():
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                shutil.move(p, real_dir / p.name)

        n = 0  # Counter
        for img_path in real_dir.rglob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            img_tensor = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0)
            
            # Input to VAE should be in [-1, 1] range (currently in [0, 1])
            img_tensor = (img_tensor * 2 - 1).to(device, torch.float32)

            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample()
                reconstructed = vae.decode(latent).sample

            # Convert back to [0, 1] range for saving
            reconstructed = ((reconstructed + 1) / 2).clamp(0, 1).cpu()
            to_pil(reconstructed.squeeze()).save(fake_dir / img_path.name, quality=95)

            n += 1
            if n % 10 == 0:
                print(f"[{cls_dir.name}] {n} done")
                
                if n % 500 == 0:
                  if device.type == "cuda":
                      torch.cuda.empty_cache()
                  elif device.type == "mps":
                      torch.mps.empty_cache()
                      
        print(f"{cls_dir.name}: {n} reconstructions")

if __name__ == "__main__":
    process_split(pathlib.Path("data/imagenette/train"))
    process_split(pathlib.Path("data/imagenette/valid"))
    print("All reconstructions finished.")
    