import pathlib, random, shutil
root = pathlib.Path("data/imagenette")
train_r = root / "train" / "real"
valid_r = root / "valid" / "real"
train_r.mkdir(parents=True, exist_ok=True)
valid_r.mkdir(parents=True, exist_ok=True)

for cls in (root / "train").iterdir():
    imgs = list(cls.glob("*.JPEG"))
    random.shuffle(imgs)
    split = int(0.9 * len(imgs))
    for p in imgs[:split]:
        (train_r / p.parent.name).mkdir(exist_ok=True)
        shutil.move(p, train_r / p.parent.name / p.name)
    for p in imgs[split:]:
        (valid_r / p.parent.name).mkdir(exist_ok=True)
        shutil.move(p, valid_r / p.parent.name / p.name)
