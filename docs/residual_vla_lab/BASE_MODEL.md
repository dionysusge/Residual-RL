# Base model provenance

| Field | Value |
|---|---|
| Source | Physical Intelligence OpenPI / RLinf-supported PyTorch model |
| Official checkpoint | `gs://openpi-assets/checkpoints/pi05_libero` |
| RLinf checkpoint option | `https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-SFT` |
| Config name | `pi05_libero` |
| OpenPI reference revision | `15a9616a00943ada6c20a0f158e3adb39df2ccac` |
| Normalization stats | supplied by the selected checkpoint/config; record exact artifact after download |
| Action horizon/chunk | RLinf v0 smoke executes H=5 |
| Action dimension | 7 |
| Observation contract | 256×256 main image, wrist image, task text, robot state |
| Official score | 96.85% mean; OpenPI `examples/libero/README.md` |
| Reproduced score | PENDING two-GPU server screening |
| Download revision/hash | PENDING artifact download; must be recorded before training |

No SFT is performed by this project. `PI05_LIBERO_CHECKPOINT` must point to a
single explicitly chosen artifact; silent checkpoint substitution is forbidden.

