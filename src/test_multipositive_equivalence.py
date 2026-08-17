"""Gradient-equivalence check: does exact-text multi-positive differ from single-positive?

With K identical text vectors in a batch, s the anchor's logit against that text and R the
summed exponentials over the distinct texts,

    L_single = -s + log(K e^s + R)
    L_multi  = -log( K e^s / (K e^s + R) ) = L_single - log K

so the two differ by a constant that does not depend on any parameter, and their gradients
are identical. The text encoder is frozen and identical texts give identical vectors, so
the model cannot push them apart from one another either. This asserts both facts on real
tensors rather than trusting the algebra.

Consequence if it passes: exact-text multi-positive is a mathematical sanity check, not a
second arm, and it does not get five seeds of its own.
"""
import numpy as np
import torch

torch.manual_seed(0)
N, D, K, TAU = 32, 64, 6, 0.07

proj = torch.nn.Linear(768, D)
audio = torch.randn(N, 768)
text = torch.randn(N, D)
text[:K] = text[0]                      # K identical schema texts, as the frozen encoder gives
text = torch.nn.functional.normalize(text, dim=1)

def logits():
    z = torch.nn.functional.normalize(proj(audio), dim=1)
    return z @ text.T / TAU

same = (text @ text.T > 1 - 1e-6)       # identical-text mask

L1 = torch.nn.functional.cross_entropy(logits(), torch.arange(N))          # single-positive
lg = logits()
L2 = -(torch.logsumexp(lg.masked_fill(~same, -1e9), 1) - torch.logsumexp(lg, 1)).mean()

proj.zero_grad(); L1.backward(retain_graph=False)
g1 = torch.cat([p.grad.flatten().clone() for p in proj.parameters()])
proj.zero_grad(); L2.backward()
g2 = torch.cat([p.grad.flatten().clone() for p in proj.parameters()])

diff = float((L1 - L2).item())
expect = float(np.log(K) * K / N)        # only the K anchors carry the -log K term
print(f"L_single {L1.item():.6f}   L_multi {L2.item():.6f}   diff {diff:.6f}")
print(f"expected mean diff (K={K} anchors of N={N}): {expect:.6f}")
print(f"max |grad_single - grad_multi| = {float((g1-g2).abs().max()):.3e}")
print(f"cosine(grad_single, grad_multi) = {float(torch.nn.functional.cosine_similarity(g1,g2,dim=0)):.10f}")
ok = abs(diff - expect) < 1e-5 and float((g1-g2).abs().max()) < 1e-5
print("\nEQUIVALENT — exact-text multi-positive is a sanity check, not an arm." if ok
      else "\nNOT EQUIVALENT — the assumption fails; multi-positive must become a real arm.")
