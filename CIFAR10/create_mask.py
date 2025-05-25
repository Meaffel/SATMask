#!/usr/bin/env python3
import numpy as np
import torch
from pysmt.shortcuts import Symbol, And, GE, LE, Plus, Times, Ite, Real, Solver
from pysmt.typing    import REAL
from torchvision import datasets, transforms
from model import SimpleCNN
import matplotlib.pyplot as plt
from pathlib import Path
import time

def save_adversarial(x0, M, y0, t, mu, sigma, eps_pixel,
                     out_dir="adv_results", prefix="example"):
    """
    x0:      (1,32,32) normalized original
    M:       (1,32,32) SMT-found normalized perturbation
    mu,sigma:    normalization params
    eps_pixel: max pixel-space L radius
    """
    out = Path(out_dir); out.mkdir(exist_ok=True)
    # recover raw pixel values in [0,1]
    x0_raw   = x0 * sigma + mu
    xadv_raw = np.clip(x0_raw + M * sigma, 0,1)
    x0_raw   = np.clip(x0_raw, 0,1)
    orig = (x0_raw[0]*255).round().astype(np.uint8)
    adv  = (xadv_raw[0]*255).round().astype(np.uint8)
    # build a mask heatmap from pixel‐space perturbation
    mask_pixel = M * sigma
    mask_norm  = np.clip((mask_pixel + eps_pixel)/(2*eps_pixel), 0,1)[0]
    mask_rgb   = (plt.cm.hot(mask_norm)*255).astype(np.uint8)[...,:3]

    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(orig, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(mask_rgb);        axes[1].set_title("Perturbation")
    axes[2].imshow(adv,  cmap='gray'); axes[2].set_title("Adversarial")
    for ax in axes: ax.axis("off")

    fn = out/f"{prefix}_cmp_{y0}_to_{t}_{time.time()}.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("Saved figure to", fn)


# 1) Load model & extract weights
device = torch.device("cpu")
model = SimpleCNN(in_channels=1, base_ch=1).to(device)
model.load_state_dict(torch.load("cifar10_gray_boat_vs_frog.pth",
                                 map_location=device))
model.eval()

wconv = model.conv.weight.detach().cpu().numpy()  # (1,1,3,3)
bconv = model.conv.bias  .detach().cpu().numpy()  # (1,)
wfc   = model.fc.weight  .detach().cpu().numpy()  # (1,16)
bfc   = model.fc.bias    .detach().cpu().numpy()  # (1,)


# 2) Grab one correctly classified gray‐scale boat(8)/frog(6)
CIFAR10_MEAN_RGB = (0.4914,0.4822,0.4465)
CIFAR10_STD_RGB  = (0.2023,0.1994,0.2010)
mean_gray = sum(CIFAR10_MEAN_RGB)/3.0   # ≈0.4733
std_gray  = sum(CIFAR10_STD_RGB )/3.0   # ≈0.2009

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((mean_gray,),(std_gray,)),
])
ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
label_map = {8:0, 6:1}
for img, lab in ds:
    if lab not in label_map: continue
    inp = img.unsqueeze(0).to(device)
    with torch.no_grad():
        # single‐logit: class=1 iff logit>0
        pred = int(model(inp).item() > 0)
    if pred == label_map[lab]:
        x0 = img.numpy()   # shape (1,32,32), normalized
        y0 = lab
        print(f"Chose correct {('boat' if lab==8 else 'frog')} (lab={lab}, pred={pred})")


        # 3) Build SMT mask vars m_{i,j} in normalized‐space
        def C(x): return Real(float(x))

        eps_pixel = 0.05           # max ±0.03 in [0,1] pixels
        sigma      = std_gray       # normalization std
        eps_norm   = eps_pixel / sigma

        mask, bounds = {}, []
        for i in range(32):
            for j in range(32):
                m = Symbol(f"m_{i}_{j}", REAL)
                mask[(i,j)] = m
                bounds += [
                    LE(m, C( eps_norm)),
                    GE(m, C(-eps_norm)),
                ]

        def IP(i,j):
            # normalized perturbed pixel
            return Plus(C(x0[0,i,j]), mask[(i,j)])


        # 4) Average‐pool 32->4 (kernel=8,stride=8)
        K = 8
        P = 32 // K                 # =4
        norm = C(1.0/(K*K))
        y_pool = {}
        for p in range(P):
            for q in range(P):
                acc = C(0)
                for di in range(K):
                    for dj in range(K):
                        ii, jj = p*K+di, q*K+dj
                        acc = Plus(acc, IP(ii,jj))
                y_pool[(p,q)] = Times(norm, acc)


        # 5) Conv3×3 + ReLU -> (1,4,4)
        C_out,C_in,Kh,Kw = wconv.shape
        pad = Kh//2
        z = {}; y = {}
        for k in range(C_out):
            for i in range(P):
                for j in range(P):
                    acc = C(0)
                    for c in range(C_in):
                        for u in range(Kh):
                            for v in range(Kw):
                                ii = i + u - pad
                                jj = j + v - pad
                                if 0 <= ii < P and 0 <= jj < P:
                                    acc = Plus(acc,
                                            Times(C(wconv[k,c,u,v]),
                                                    y_pool[(ii,jj)]))
                    acc = Plus(acc, C(bconv[k]))
                    z[(k,i,j)] = acc
                    y[(k,i,j)] = Ite(GE(acc, C(0)), acc, C(0))


        # 6) Flatten & FC -> single logit
        flat = [ y[(0,i,j)] for i in range(P) for j in range(P) ]  # len=16
        acc = C(0)
        for idx, f in enumerate(flat):
            acc = Plus(acc, Times(C(wfc[0,idx]), f))
        logit = Plus(acc, C(bfc[0]))


        # 7) Misclassification constraint
        y0_cd = label_map[y0]       # 0 or 1
        t     = 1 - y0_cd           # target flipped class
        delta     = C(1e-3)

        if t == 1:
            # want logit >= +delta -> class 1
            mis = GE(logit, delta)
        else:
            # want logit <= -delta -> class 0
            mis = LE(logit, C(-delta.constant_value()))


        # 8) Solve with Z3
        formula = And(*(bounds + [mis]))
        with Solver(name="z3") as solver:
            solver.add_assertion(formula)
            if solver.solve():
                print("FOUND ADVERSARIAL PERTURBATION")
                M = np.zeros((1,32,32),dtype=np.float32)
                for (i,j), sym in mask.items():
                    M[0,i,j] = float(solver.get_value(sym).constant_value())

                xadv = x0 + M
                x0_t  = torch.from_numpy(x0).unsqueeze(0)
                xadv_t= torch.from_numpy(xadv).unsqueeze(0)
                with torch.no_grad():
                    p0 = int(model(x0_t).item() > 0)
                    p1 = int(model(xadv_t).item() > 0)
                print(f"orig->{p0}, adv->{p1} (target {t})")

                save_adversarial(x0, M, y0, t,
                                mu=mean_gray, sigma=std_gray, eps_pixel=eps_pixel)
            else:
                print("UNSAT: no perturbation fools the net.")