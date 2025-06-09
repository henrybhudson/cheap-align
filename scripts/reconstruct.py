import pathlib, shutil, torch, torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL

vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to("mps").eval()
vae.enable_tiling()
SCALE = vae.config.scaling_factor          # 0.18215

to_tensor = T.Compose([T.Resize(512, antialias=True), T.ToTensor()])
to_pil    = T.ToPILImage()

def process_split(split_root: pathlib.Path):
    """
    split_root = data/imagenette/train   or   .../valid
    """
    classes = [d for d in split_root.iterdir() if d.is_dir()]
    for cls_dir in classes:
        real_dir = cls_dir / "0_real"
        fake_dir = cls_dir / "1_fake"
        real_dir.mkdir(parents=True, exist_ok=True)
        fake_dir.mkdir(parents=True, exist_ok=True)

        for img_path in cls_dir.glob("*.[jpJP][pnN]*"):
            shutil.move(str(img_path), real_dir / img_path.name)

        n = 0
        for img_path in real_dir.rglob("*"):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            # load → [-1,1] → mps
            x = to_tensor(Image.open(img_path).convert("RGB")).unsqueeze(0)
            x = (x * 2 - 1).to("mps")

            with torch.no_grad():
                lat = vae.encode(x).latent_dist.sample() * SCALE
                rec = vae.decode(lat / SCALE).sample

            rec = ((rec + 1) / 2).clamp(0, 1).cpu()
            to_pil(rec.squeeze()).save(
                fake_dir / img_path.name, quality=95
            )

            n += 1
            if n % 500 == 0:
                print(f"[{cls_dir.name}] {n} images done")
                torch.mps.empty_cache()

        print(f"{cls_dir.name}: {n} reconstructions")

if __name__ == "__main__":
    process_split(pathlib.Path("data/imagenette/train"))
    process_split(pathlib.Path("data/imagenette/valid"))
    print("All reconstructions finished.")