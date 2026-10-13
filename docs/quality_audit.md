# Quality Audit Findings

Fecha: 2026-10-12 · Rama: `dev` · Baseline: `89b4117`

Herramientas: `vulture`, `bandit`, `radon`, `ruff`, `mypy`.

## vulture (training/, inference/, scripts/)

| Ubicación | Hallazgo | Acción |
|---|---|---|
| `training/model.py:3` | Import sin usar `GlobalAveragePooling2D` | eliminar import |
| `training/dataset.py:118` | `on_epoch_end` (fine si se usa con Keras `Sequence`) | mantener |
| `training/evaluate.py:10,82,100` | variable `fig` sin usar | `fig, _ = ...` |
| `scripts/build_dataset.py:18` | variable de bucle `files` sin usar | `_` |
| `inference/predictor.py:15` | `confidence_threshold` se asigna pero no se usa | documentar/revisar |
| `training/model.py:55` | `base.trainable = False` reportado falso (+) | mantener |

## bandit (training/, inference/, scripts/)

0 issues (severity Low/Medium/High). Sin hallazgos de seguridad pendientes.

## radon cc

Complejidad ciclomática negra suavizada: todo grade A/B, sin grade C-F.
Puntos más complejos: `evaluate_model` (B-7), `plot_training_history` (B-6),
`load_dataset` (B-9), `process_captures` (B-6). `radon mi`: todo grade A.

## ruff check .

21 errores: I001 (orden de imports) × 18, B007 × 1, SIM105 × 1,
SIM114 × 1, F401 × 2, B017 × 1. Detalle en baseline.

## mypy .

Tras `ignore_missing_imports` quedan erros reales:
- `scripts/build_dataset.py:59` falta anotación de tipo en `added`
- `training/dataset.py:112` `ImageDataGenerator` opcional (**posible None**)
- `training/dataset.py:136-138` `*split_data(...)` + kwargs duplican `batch_size`/`augment`
- `inference/visualizer.py:17` `cv2.VideoWriter_fourcc` no tipado

## Duplicación de carga de datos (RESUELTO)

Lógica de carga CSV/metadata/decodificación de clases (config.yaml) repetida entre:
`training/dataset.py`, `training/evaluate.py`, `training/train.py`,
`inference/predictor.py`, `app.py`, `scripts/build_dataset.py`.

Se extrajo a un módulo compartido `common/` sin cambiar comportamiento:
- `common/config.py` — carga centralizada de `config.yaml` + overrides de .env
  (antes en `inference/config.py`, ahora re-exporta).
- `common/data.py` — decodificación de labels, extensión de imágenes, preproceso
  (`preprocess_image`, `preprocess_frame`), onehot, mapas de directorios.
- `training/dataset.py`, `training/evaluate.py`, `training/model.py`,
  `inference/predictor.py`, `app.py` y `scripts/build_dataset.py` importan
  desde `common` en vez de duplicar lógica.