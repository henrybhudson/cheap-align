import sys, subprocess, pathlib, argparse

BASE = pathlib.Path(__file__).resolve().parents[1]
THIRDP = BASE / "thirdparty" / "AlignedForensics"

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("--real", default="data/real_10k_aligned")
        parser.add_argument("--fake", default="data/fake_10k")
        parser.add_argument("--arch", default="efficientnet_b0")
        parser.add_argument("--epochs", type=int, default=10)
        args = parser.parse_args()
        
        cmd = [
                sys.executable,
                str(THIRDP / "train_detector.py"),
                "--real_dir", args.real,
                "--fake_dir", args.fake,
                "--arch", args.arch,
                "--epochs", str(args.epochs),
                "--batch", "64",
                "--seed", "42"
        ]
        
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        
if __name__ == "__main__":
        main()