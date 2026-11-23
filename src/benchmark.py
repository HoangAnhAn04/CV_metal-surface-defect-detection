import time

import torch

from common import MODELS, build_model

NAMES = ["cnn_baseline", "cnn_strong", "resnet18_ft", "resnet18_strong"]
WARMUP = 20
RUNS = 100


def load(name, device):
    ckpt = torch.load(MODELS / f"{name}.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt["img_size"]


@torch.no_grad()
def latency(model, size, device):
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(WARMUP):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(RUNS):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / RUNS * 1000


def main():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    head = f"{'mo hinh':<18}{'tham so':>12}{'MB':>8}"
    head += "".join(f"{'ms ' + d.type:>10}" for d in devices)
    print(head)
    for name in NAMES:
        times = []
        for d in devices:
            model, size = load(name, d)
            times.append(latency(model, size, d))
        cpu_model, _ = load(name, torch.device("cpu"))
        params = sum(p.numel() for p in cpu_model.parameters())
        mb = (MODELS / f"{name}.pt").stat().st_size / 1e6
        print(f"{name:<18}{params:>12,}{mb:>8.1f}" + "".join(f"{t:>10.2f}" for t in times))


if __name__ == "__main__":
    main()