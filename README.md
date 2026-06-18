# Lightweight Extraction

This folder is a small, independent adaptation for neighbor-retrieval
experiments. It intentionally avoids the full `timetensors` dataset,
dataloader, augmentation, normalization, and experiment stack.

## Files

- `load_dataset_model.py`: load a CSV dataset and a pretrained model.
- `models.py`: minimal model wrapper; only `none` and `instance`
  normalization are supported.
- `foundation_models.py`: Chronos and PatchTST wrappers.
- `neighbors.py`: deterministic windows, aligned datastore dates, feature
  representations, and exact KNN search.
- `extraction.py`: aligned neighbor extraction script.
- `features.py`: build flat feature tables and plots from extraction payloads.
- `experiment_univariate.py`: univariate all-user evaluation and plots.
- `visu/`: plotting helpers and notebook.

## Example

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' -m extraction.experiment_univariate `
  --csv ../datasets/electricity/electricity.csv `
  --lags 168 `
  --horizon 24 `
  --model persistence `
  --normalization none `
  --eval-stride 24 `
  --output-dir outputs/extraction_univariate `
  --save-name electricity_persistence
```

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' -m extraction.extraction `
  --csv ../datasets/electricity/electricity.csv `
  --lags 168 `
  --horizon 24 `
  --model chronos `
  --model-kwargs '{"weights_path":"path/to/chronos/weights","context_mode":"past_only"}' `
  --neighbors 5 `
  --distance-space chronos `
  --pool-representation `
  --distance-metric cosine `
  --train-stride 24 `
  --eval-stride 24 `
  --period 24 `
  --output-dir outputs/extraction_neighbors `
  --save-name electricity_chronos_k5
```

Then:

```powershell
& 'C:\Users\Gaspard\AppData\Local\Programs\Python\Python313\python.exe' -m extraction.features `
  --input-dir outputs/extraction_neighbors/electricity_chronos_k5
```
