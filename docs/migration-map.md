# Migration Map: Legacy Scripts to New Modules

The original SplitFed reference codebase (Thapa et al. 2022) is archived under `old/`. Only the FL-relevant components were extracted into the new modular layout.

## Legacy Files (in `old/`)

| File | Description | Used in new code? |
|------|-------------|-------------------|
| `FL_ResNet_HAM10000.py` | FL single-process simulation | Yes — extracted |
| `Normal_ResNet_HAM10000.py` | Centralized baseline | No |
| `SL_ResNet_HAM10000.py` | Split Learning | No (out of scope) |
| `SFLV1_ResNet_HAM10000.py` | SplitFed V1 | No (out of scope) |
| `SFLV2_ResNet_HAM10000.py` | SplitFed V2 | No (out of scope) |

## Extraction Map

| New module | Extracted from | What |
|------------|---------------|------|
| `src/data/dataset.py` | `FL_ResNet_HAM10000.py` | `SkinData`, `DatasetSplit` |
| `src/data/partitioning.py` | `FL_ResNet_HAM10000.py` | `dataset_iid()` |
| `src/data/preprocessing.py` | `FL_ResNet_HAM10000.py` | Transforms, train/test split, lesion mapping |
| `src/models/resnet.py` | `FL_ResNet_HAM10000.py` | `BasicBlock`, `ResNet18` |
| `src/fl/aggregation.py` | `FL_ResNet_HAM10000.py` | `FedAvg` |
| `src/fl/client_core.py` | `FL_ResNet_HAM10000.py` | `LocalUpdate` (refactored as `LocalTrainer`) |
| `src/fl/server_core.py` | `FL_ResNet_HAM10000.py` | Main loop (refactored as `FLServer`) |
| `src/training/metrics.py` | `FL_ResNet_HAM10000.py` | `calculate_accuracy` |
| `src/transport/interface.py` | New | Transport abstraction |
| `src/transport/http_adapter.py` | New | HTTP+JSON adapter |
| `src/transport/serialization.py` | New | state_dict bytes roundtrip |
