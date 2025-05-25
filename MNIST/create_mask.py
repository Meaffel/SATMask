#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pysmt.shortcuts import Symbol, And, GE, LE, Plus, Times, Ite, Real, Solver
from pysmt.typing    import REAL

from torchvision import datasets, transforms
from model import SimpleCNN   # your new MNIST model

# -----------------------------------------------------------------------------
def save_adversarial(x0, M, y0, t, mu, sigma, eps_pixel,
                     out_dir="adv_results", prefix="mnist_adv"):
    """
    x0:      (1,28,28) normalized original
    M:       (1,28,28) normalized perturbation
    mu,sigma:    normalization params
    eps_pixel: max pixel-space L radius
    """
    out = Path(out_dir); out.mkdir(exist_ok=True)
    # recover raw [0,1]
    x0_raw   = x0 * sigma + mu
    xadv_raw = np.clip(x0_raw + M * sigma, 0,1)
    x0_raw   = np.clip(x0_raw, 0,1)
    # to uint8
    orig = (x0_raw[0]*255).round().astype(np.uint8)
    adv  = (xadv_raw[0]*255).round().astype(np.uint8)

    # mask heatmap
    mask_pixel = M * sigma
    mask_norm  = np.clip((mask_pixel + eps_pixel)/(2*eps_pixel),0,1)[0]
    mask_rgb   = (plt.cm.hot(mask_norm)*255).astype(np.uint8)[...,:3]

    fig, axes = plt.subplots(1,3,figsize=(12,4))
    axes[0].imshow(orig, cmap='gray'); axes[0].set_title("Original")
    axes[1].imshow(mask_rgb);        axes[1].set_title("Perturbation")
    axes[2].imshow(adv,  cmap='gray'); axes[2].set_title("Adversarial")
    for ax in axes: ax.axis("off")

    fn = out/f"{prefix}_cmp_{y0}_to_{t}_{int(time.time())}.png"
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print("Saved figure to", fn)

# -----------------------------------------------------------------------------
# 1) Load model & extract weights
device = torch.device("cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn_strided.pth", map_location=device))
model.eval()

wconv = model.conv.weight .detach().cpu().numpy()  # (1,1,3,3)
bconv = model.conv.bias   .detach().cpu().numpy()  # (1,)
wfc   = model.fc.weight   .detach().cpu().numpy()  # (1,16)
bfc   = model.fc.bias     .detach().cpu().numpy()  # (1,)

# -----------------------------------------------------------------------------
# 2) Grab one correctly classified MNIST digit 0 or 1
mu, sigma = 0.1307, 0.3081
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mu,),(sigma,)),
])
ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
for img, lab in ds:
    if lab not in (0,1): continue
    inp = img.unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(inp).item()
        pred  = int(logit > 0)
    if pred == lab:
        x0 = img.numpy()   # (1,28,28)
        y0 = lab
        print(f"Picked MNIST {y0}, model predicted {pred}.")

        # -----------------------------------------------------------------------------
        # 3) SMT mask vars in normalized space
        def C(x): return Real(float(x))

        eps_pixel = 0.05           # max ±0.3 in raw [0,1]
        eps_norm  = eps_pixel / sigma

        H,W = 28,28
        mask, bounds = {}, []
        for i in range(H):
            for j in range(W):
                m = Symbol(f"m_{i}_{j}", REAL)
                mask[(i,j)] = m
                bounds += [ LE(m, C(+eps_norm)), GE(m, C(-eps_norm)) ]

        def IP(i,j):
            return Plus(C(x0[0,i,j]), mask[(i,j)])

        # -----------------------------------------------------------------------------
        # 4) AvgPool 28->4 (kernel=7,stride=7)
        K = 7
        P = H // K   # =4
        norm = C(1.0/(K*K))
        y_pool = {}
        for p in range(P):
            for q in range(P):
                acc = C(0)
                for di in range(K):
                    for dj in range(K):
                        ii,jj = p*K+di, q*K+dj
                        acc = Plus(acc, IP(ii,jj))
                y_pool[(p,q)] = Times(norm, acc)

        # -----------------------------------------------------------------------------
        # 5) Conv3×3 + ReLU -> 1×4×4
        pad = 1
        z = {}; y = {}
        for i in range(P):
            for j in range(P):
                acc = C(0)
                for u in range(3):
                    for v in range(3):
                        ii, jj = i+u-pad, j+v-pad
                        if 0 <= ii < P and 0 <= jj < P:
                            acc = Plus(acc,
                                    Times(C(wconv[0,0,u,v]),
                                            y_pool[(ii,jj)]))
                z[(i,j)] = Plus(acc, C(bconv[0]))
                y[(i,j)] = Ite(GE(z[(i,j)], C(0)), z[(i,j)], C(0))

        # -----------------------------------------------------------------------------
        # 6) Flatten & single‐logit FC
        flat = [y[(i,j)] for i in range(P) for j in range(P)]  # len=16
        acc  = C(0)
        for idx,f in enumerate(flat):
            acc = Plus(acc, Times(C(wfc[0,idx]), f))
        logit = Plus(acc, C(bfc[0]))

        # -----------------------------------------------------------------------------
        # 7) Misclassification: flip sign of logit
        t = 1 - y0
        delta = C(1e-3)
        if t == 1:
            mis = GE(logit, delta)      # want logit >= +delta -> class 1
        else:
            mis = LE(logit, C(-delta.constant_value()))  # logit <= -delta -> class 0

        # -----------------------------------------------------------------------------
        # 8) Solve
        formula = And(*(bounds + [mis]))
        with Solver(name="z3") as solver:
            solver.add_assertion(formula)
            if solver.solve():
                print("FOUND ADV PERTURBATION")
                M = np.zeros((1,H,W), dtype=np.float32)
                for (i,j), sym in mask.items():
                    M[0,i,j] = float(solver.get_value(sym).constant_value())

                # verify
                xadv = x0 + M
                with torch.no_grad():
                    p0 = int(model(torch.from_numpy(x0).unsqueeze(0)).item() > 0)
                    p1 = int(model(torch.from_numpy(xadv).unsqueeze(0)).item() > 0)
                print(f"orig->{p0}, adv->{p1} (target {t})")

                save_adversarial(x0, M, y0, t,
                                mu=mu, sigma=sigma, eps_pixel=eps_pixel)
            else:
                print("UNSAT: no PERTURBATION fools the net")