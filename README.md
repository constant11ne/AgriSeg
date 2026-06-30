# AgriSeg: эксперименты с RGB/NIR/NDVI/GVI-сегментацией

Репозиторий содержит эксперименты по семантической сегментации сельскохозяйственных аномалий на мультиспектральных снимках полей. Основной вопрос проекта — как лучше объединять спектральную информацию (`NIR`, `NDVI`, `GVI`) с RGB-признаками в сегментационной модели.

Текущий лучший результат получен двухветочной моделью **RGB + обучаемый GVI** с объединением признаков на среднем уровне и IBN-выравниванием:

```text
mid_rgb_gvi_ibn: mIoU = 0.5532
```

Этот результат лучше предыдущей лучшей модели с фиксированным спектральным индексом:

```text
mid_rgb_ndvi_ibn: mIoU = 0.5418
```

Эксперименты показывают, что **обучаемый vegetation index** оказался полезнее фиксированного NDVI-канала. При этом простое добавление нескольких GVI-каналов сразу в текущей конфигурации качества не улучшило.

---

## 1. Обзор проекта

Проект посвящён семантической сегментации на RGB+NIR аэроснимках сельскохозяйственных полей. Каждый входной снимок содержит каналы:

```text
R, G, B, NIR
```

Целевая разметка — multilabel segmentation mask для классов сельскохозяйственных аномалий:

```text
double_plant, drydown, endrow, nutrient_deficiency,
planter_skip, storm_damage, water, waterway, weed_cluster
```

Основное семейство архитектур:

```text
FPN + EfficientNet-B4 encoder
```

Основная функция потерь в финальных экспериментах:

```text
soft_bce_dice
bce_weight  = 0.12
dice_weight = 0.88
dice_smooth = 1e-5
```

---

## 2. Основная идея экспериментов

Первые эксперименты показали, что простое добавление дополнительных каналов не всегда помогает. Лучшим направлением стало раздельное извлечение признаков из RGB и спектрального индекса:

```text
RGB branch + spectral-index branch -> mid-level fusion -> FPN decoder
```

Фиксированный индекс `NDVI` оказался полезен, если обрабатывать его как отдельную ветку. Следующим шагом стала замена фиксированного NDVI на обучаемый generalized vegetation index:

```text
GVI = learned_num(R, G, B, NIR) / (learned_den(R, G, B, NIR) + eps)
```

На практике GVI-модуль инициализируется так, чтобы сначала вести себя похоже на NDVI, а затем дообучается end-to-end вместе с сегментационной моделью.

---

## 3. Основные волны экспериментов

### 3.1 NDVI wave

В этой волне проверялся фиксированный NDVI: как дополнительный входной канал и как отдельная ветка.

| Эксперимент | Описание | Best mIoU | Best epoch | Last mIoU |
|---|---:|---:|---:|---:|
| `ndvi_only` | только фиксированный NDVI | 0.4340 | 35 | 0.4340 |
| `early_rgb_ndvi` | раннее объединение RGB+NDVI | 0.4701 | 14 | 0.4068 |
| `mid_rgb_ndvi` | двухветочная модель RGB+NDVI | 0.5343 | 32 | 0.5259 |
| `mid_rgb_ndvi_ibn` | RGB+NDVI с IBN-выравниванием | 0.5418 | 32 | 0.5360 |

Главный вывод: фиксированный NDVI полезен, если использовать его как отдельную ветку и объединять признаки на feature-level. IBN-выравнивание даёт дополнительный прирост.

---

### 3.2 Single-GVI wave

В этой волне фиксированный NDVI был заменён на обучаемый GVI-модуль.

| Эксперимент | Описание | Best mIoU | Best epoch | Last mIoU | Macro F1 | Rare mIoU |
|---|---|---:|---:|---:|---:|---:|
| `gvi_only` | модель только на GVI, инициализация от `ndvi_only` | 0.4887 | 35 | 0.4887 | 0.6433 | 0.4968 |
| `early_rgb_gvi` | раннее объединение RGB+GVI | 0.5470 | 35 | 0.5470 | 0.6975 | 0.5527 |
| `mid_rgb_gvi` | двухветочная RGB+GVI модель | 0.5442 | 35 | 0.5442 | 0.6953 | 0.5473 |
| `mid_rgb_gvi_ibn` | двухветочная RGB+GVI модель с IBN | **0.5532** | 35 | **0.5532** | **0.7040** | **0.5573** |

Основные наблюдения:

1. `gvi_only` заметно сильнее, чем `ndvi_only`:

```text
0.4887 vs 0.4340, gain = +0.0548 mIoU
```

2. `early_rgb_gvi` оказался очень сильным baseline и обошёл `mid_rgb_ndvi_ibn`:

```text
early_rgb_gvi:    0.5470
mid_rgb_ndvi_ibn: 0.5418
```

3. Лучшей моделью остаётся двухветочная версия с IBN:

```text
mid_rgb_gvi_ibn: 0.5532
```

4. `mid_rgb_gvi_ibn` улучшает результат относительно `mid_rgb_ndvi_ibn`:

```text
mIoU:      +0.0113
macro F1:  +0.0097
rare mIoU: +0.0116
```

5. Прирост не сосредоточен в одном классе: улучшение положительное для всех классов.

| Класс | Прирост IoU относительно `mid_rgb_ndvi_ibn` |
|---|---:|
| `double_plant` | +0.0092 |
| `drydown` | +0.0069 |
| `endrow` | +0.0103 |
| `nutrient_deficiency` | +0.0031 |
| `planter_skip` | +0.0077 |
| `storm_damage` | +0.0214 |
| `water` | +0.0069 |
| `waterway` | +0.0188 |
| `weed_cluster` | +0.0176 |

Интерпретация: замена фиксированного NDVI на обучаемый GVI-модуль даёт устойчивый прирост. Модель выигрывает от того, что спектральное преобразование подстраивается под задачу, а не задаётся одной фиксированной формулой.

---

### 3.3 Multi-GVI wave

В этой волне проверялось, полезно ли использовать несколько обучаемых GVI-каналов вместо одного.

Все модели использовали схему:

```text
RGB branch + Multi-GVI branch + IBN alignment
```

Вторая ветка содержала `K` GVI-каналов, инициализированных разными формулами индексов.

| Эксперимент | GVI channels | Инициализация | Best mIoU | Best epoch | Last mIoU | Macro F1 | Rare mIoU |
|---|---:|---|---:|---:|---:|---:|---:|
| `mid_rgb_multigvi2_ibn` | 2 | NDVI, GNDVI | 0.5330 | 34 | 0.4924 | 0.6866 | 0.5368 |
| `mid_rgb_multigvi3_ibn` | 3 | NDVI, GNDVI, NDWI | 0.5122 | 16 | 0.4748 | 0.6656 | 0.5188 |
| `mid_rgb_multigvi4_ibn` | 4 | NDVI, GNDVI, NDWI, random | 0.5168 | 21 | 0.4365 | 0.6721 | 0.5231 |

Основные наблюдения:

1. Multi-GVI не улучшил результат относительно single-GVI.

```text
single GVI + IBN: 0.5532
best Multi-GVI:   0.5330
```

2. Лучший вариант Multi-GVI — `K=2`, то есть инициализация `NDVI + GNDVI`.

3. Увеличение числа GVI-каналов делает обучение менее стабильным:

```text
K=2: best 0.5330 -> last 0.4924
K=3: best 0.5122 -> last 0.4748
K=4: best 0.5168 -> last 0.4365
```

4. Текущая Multi-GVI-конфигурация, вероятно, страдает от более сложной оптимизации и избыточности спектральных каналов. Увеличение числа каналов нагружает вторую ветку и fusion-блок, но не даёт дополнительной пользы.

Интерпретация: в текущей архитектуре один обучаемый GVI-канал работает лучше, чем набор из нескольких обучаемых индексов. Больше спектральных каналов не означает автоматически лучшее качество.

---

## 4. Текущий leaderboard

| Rank | Model | Best mIoU | Best epoch | Last mIoU | Notes |
|---:|---|---:|---:|---:|---|
| 1 | `mid_rgb_gvi_ibn` | **0.5532** | 35 | **0.5532** | лучший результат |
| 2 | `early_rgb_gvi` | 0.5470 | 35 | 0.5470 | сильный early-GVI baseline |
| 3 | `mid_rgb_gvi` | 0.5442 | 35 | 0.5442 | сильная двухветочная модель без IBN |
| 4 | `mid_rgb_ndvi_ibn` | 0.5418 | 32 | 0.5360 | лучший результат среди fixed-index моделей |
| 5 | `mid_rgb_ndvi` | 0.5343 | 32 | 0.5259 | фиксированный NDVI как отдельная ветка |
| 6 | `mid_rgb_multigvi2_ibn` | 0.5330 | 34 | 0.4924 | лучший Multi-GVI, нестабильный |
| 7 | `progressive RGB+NIR baseline` | 0.5181 | 28 | 0.4976 | лучший baseline без NDVI/GVI |
| 8 | `mid_rgb_multigvi4_ibn` | 0.5168 | 21 | 0.4365 | нестабильный |
| 9 | `mid_rgb_multigvi3_ibn` | 0.5122 | 16 | 0.4748 | нестабильный |
| 10 | `gvi_only` | 0.4887 | 35 | 0.4887 | лучше, чем NDVI-only |
| 11 | `rgb_only` | 0.4724 | 16 | 0.3964 | RGB baseline |
| 12 | `ndvi_only` | 0.4340 | 35 | 0.4340 | fixed-index baseline |

---

## 5. Основные выводы

### 5.1 GVI лучше фиксированного NDVI

Модель только на обучаемом GVI заметно превосходит модель только на фиксированном NDVI:

```text
gvi_only:  0.4887
ndvi_only: 0.4340
```

Это подтверждает, что модель выигрывает от обучения task-specific vegetation index.

### 5.2 GVI полезен и при early fusion, и как отдельная ветка

`early_rgb_gvi` показывает сильный результат:

```text
early_rgb_gvi: 0.5470
```

Это важно: даже простое раннее объединение RGB с обучаемым индексом оказывается конкурентоспособным.

### 5.3 Лучшая конфигурация — двухветочная fusion-модель с IBN

Самая сильная модель:

```text
RGB branch + GVI branch + IBN alignment
```

с результатом:

```text
mIoU = 0.5532
```

### 5.4 Multi-GVI пока не помогает

Добавление нескольких GVI-каналов не улучшило качество. Возможные причины:

- избыточность индексных каналов;
- отсутствие отдельного pretraining-этапа для Multi-GVI;
- более сложная оптимизация спектральной ветки;
- более сильное расхождение признаков перед fusion;
- недостаточная регуляризация или отсутствие gating между GVI-каналами.

---

## 6. Как запускать эксперименты

### 6.1 Full GVI wave

Запускаемые эксперименты:

```text
gvi_only
early_rgb_gvi
mid_rgb_gvi
mid_rgb_gvi_ibn
```

Команда полного запуска:

```bash
python -m notebooks.experiments_gvi \
  --processed-root ../AgricultureVision_baseline_3fold \
  --encoder timm-efficientnet-b4 \
  --model fpn \
  --epochs 35 \
  --batch-size 16 \
  --img-size 256 256 \
  --runs-dir runs/gvi_wave \
  --rgb-source-runs-dir runs/augmentations_full_1fold \
  --ndvi-source-runs-dir runs/ndvi_wave2 \
  --extra-train-flags "--max-grad-norm 0.8 --num-workers 8 --prefetch-factor 4 --save-every 1000"
```

Smoke test:

```bash
python -m notebooks.experiments_gvi \
  --processed-root ../AgricultureVision_baseline_3fold \
  --encoder timm-efficientnet-b4 \
  --model fpn \
  --epochs 1 \
  --batch-size 16 \
  --img-size 256 256 \
  --runs-dir runs/gvi_wave \
  --rgb-source-runs-dir runs/augmentations_full_1fold \
  --ndvi-source-runs-dir runs/ndvi_wave2 \
  --extra-train-flags "--max-grad-norm 0.8 --num-workers 8 --prefetch-factor 4 --save-every 1000 --max-train-steps-per-epoch 10 --max-val-steps-per-epoch 5"
```

### 6.2 Multi-GVI wave

Запускаемые эксперименты:

```text
mid_rgb_multigvi2_ibn
mid_rgb_multigvi3_ibn
mid_rgb_multigvi4_ibn
```

Команда полного запуска:

```bash
python -m notebooks.experiments_multi_gvi \
  --processed-root ../AgricultureVision_baseline_3fold \
  --encoder timm-efficientnet-b4 \
  --model fpn \
  --epochs 35 \
  --batch-size 16 \
  --img-size 256 256 \
  --runs-dir runs/multi_gvi_wave \
  --rgb-source-runs-dir runs/augmentations_full_1fold \
  --extra-train-flags "--max-grad-norm 0.8 --num-workers 8 --prefetch-factor 4 --save-every 1000"
```

Smoke test:

```bash
python -m notebooks.experiments_multi_gvi \
  --processed-root ../AgricultureVision_baseline_3fold \
  --encoder timm-efficientnet-b4 \
  --model fpn \
  --epochs 1 \
  --batch-size 16 \
  --img-size 256 256 \
  --runs-dir runs/multi_gvi_wave \
  --rgb-source-runs-dir runs/augmentations_full_1fold \
  --extra-train-flags "--max-grad-norm 0.8 --num-workers 8 --prefetch-factor 4 --save-every 1000 --max-train-steps-per-epoch 10 --max-val-steps-per-epoch 5"
```

---

## 7. Структура кода

```text
src/train.py
    Основная точка входа для обучения. Поддерживает RGB, NIR, NDVI, GVI, Multi-GVI, early/mid/late fusion.

src/datamodules/agri_vision.py
    Dataset и dataloader logic. Обрабатывает RGB, NIR, NDVI и augmentations.

src/models/gvi.py
    LearnableRatioGVI и Multi-GVI модули.

src/models/multi_stream_fpn.py
    Multi-branch FPN для объединения RGB, NIR, NDVI и GVI признаков.

src/models/dual_stream_fpn.py
    Более старая dual-stream RGB/NIR FPN реализация.

notebooks/experiments_gvi.py
    Полная single-GVI wave.

notebooks/experiments_multi_gvi.py
    Multi-GVI wave с 2/3/4 каналами спектральных индексов.

notebooks/experiments_ndvi_wave2.py
    Эксперименты с фиксированным NDVI branch.

notebooks/experiments_ndvi_wave2_ibn.py
    RGB+NDVI+IBN эксперимент.
```

---

## 8. Рекомендуемые следующие шаги

Дальше лучше не просто добавлять новые raw-каналы, а стабилизировать и проверить лучший GVI-результат.

Рекомендуемый порядок:

1. Запустить `mid_rgb_gvi_ibn` на 3 seeds или 3 folds.
2. Сравнить с `mid_rgb_ndvi_ibn` на тех же seeds/folds.
3. Добавить threshold tuning по классам.
4. Добавить TTA для `mid_rgb_gvi_ibn`.
5. Попробовать ensemble из моделей:

```text
mid_rgb_gvi_ibn
early_rgb_gvi
mid_rgb_ndvi_ibn
```

6. Если возвращаться к Multi-GVI, не стоит просто увеличивать число каналов. Лучше попробовать:

```text
multi_gvi_only pretraining
lower LR for GVI module
channel attention over GVI channels
orthogonality/decorrelation loss between GVI channels
freeze GVI channels for first few epochs
```

---

## 9. Краткий итог

Лучший результат на текущем наборе экспериментов:

```text
RGB + GVI + IBN: mIoU = 0.5532
```

Лучший результат с фиксированным NDVI:

```text
RGB + NDVI + IBN: mIoU = 0.5418
```

Прирост:

```text
+0.0113 mIoU
```

Основной практический вывод: обучаемый GVI полезнее фиксированного NDVI, особенно в двухветочной модели с IBN-выравниванием. При этом Multi-GVI в текущей реализации не дал прироста и требует отдельной доработки.
