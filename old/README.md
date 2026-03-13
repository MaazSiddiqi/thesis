# Archived Legacy Scripts

These files are the original single-process simulation scripts from the
[SplitFed reference codebase](https://arxiv.org/pdf/2004.12088.pdf) (Version 1:
without sockets, no DP+PixelDP).

They are preserved for reference and comparison. The new modular implementation
lives under `src/`. See `docs/phase-01/migration-map.md` for the mapping from
these files to the new modules.

## Files

| File | Description |
|------|-------------|
| `FL_ResNet_HAM10000.py` | Federated Learning (single-process simulation) |
| `Normal_ResNet_HAM10000.py` | Centralized baseline |
| `SL_ResNet_HAM10000.py` | Split Learning (single-process) |
| `SFLV1_ResNet_HAM10000.py` | SplitFed V1 (single-process) |
| `SFLV2_ResNet_HAM10000.py` | SplitFed V2 (single-process) |
| `Normal_Learning_ResNet18_on_HAM10000_(TEST_-_1_epoch).xlsx` | Sample output from 1-epoch test |
| `SETUP.md` | Old setup instructions (pre-containerization) |
| `QUICKSTART.md` | Old quick-start guide |
