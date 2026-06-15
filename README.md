# Natural Language Description of Geospatial Relationships

A two-stage pipeline that generates fluent natural language descriptions of
spatial relationships between Points of Interest (POI), combining
deterministic geometric reasoning with a fine-tuned language model.

> *"Bisounours Excursion is north of Place Commandant Renard."*

## Overview

Generating spatial descriptions requires two very different kinds of
computation: precise geometry (direction, distance, bearings) and fluent,
varied language generation. LLMs are unreliable at coordinate arithmetic,
while hand-written templates are factually correct but repetitive.

This project separates the two concerns:

* **Stage 1** — a lightweight 3-layer MLP classifies the cardinal direction
(N/S/E/W) between two POI from their coordinates, with a confidence score.
* **Stage 2** — a QLoRA fine-tuned `Phi-3-mini-4k-instruct` model takes the
predicted direction, the Haversine distance, and the POI names, and
generates a fluent sentence.

All training and inference run on a single consumer GPU (GTX 1080 Ti, 11GB
VRAM).

## Pipeline

```
POI A, POI B (names)
  → coordinate lookup (OSM index)
  → Haversine distance
  → Stage 1 (MLP): cardinal direction + confidence
  → Stage 2 (Phi-3-mini, QLoRA): natural language sentence

```

## Dataset

* Source: OpenStreetMap (Nancy, France) via `osmnx`
* 3,950 named POI across 7 categories (amenity, shop, public transport,
office, tourism, leisure, historic)
* 1,882,147 POI pairs retained after filtering (distance ≤ 5km, direction
confidence ≥ 0.75)
* 18,000 template-generated sentences used for Stage 2 fine-tuning
* 300-pair human-verified gold standard for evaluation

## Results

| Metric | Stage 1 | Stage 2 (gold dir.) | Combined Pipeline |
| --- | --- | --- | --- |
| Direction Accuracy | 100.00% | 96.33% | 94.00% |
| Semantic Similarity | — | 93.01% | 92.90% |
| Diversity Score | — | 47.13% | 46.91% |

Stage 1's 100% accuracy reflects a deterministic geometric identity rather
than a hard learning problem; the modular interface it provides is the
point, not the score. Stage 2 is where the contribution lies: fine-tuning
on a small, carefully constructed corpus produces output that is
substantially more lexically diverse (47.13%) than a template baseline
(~30%) or the base model under zero/few-shot prompting (36.0% / 34.9%),
while remaining factually grounded.

## Model Exploration & Branches

While the final production pipeline utilizes `Phi-3-mini-4k-instruct` (maintained on the `microsoft-phi-mini3.8` branch), our team conducted extensive empirical testing across several different LLM architectures to optimize for performance, memory constraints, and carbon efficiency.

A complete experimental pipeline was built from the ground up for the **Qwen** architecture, including dedicated scripts for extracting and filtering OSM data. While these models demonstrated strong potential, they were ultimately superseded by Phi-3, which offered a superior performance-to-compute ratio, faster training times, and a lower carbon footprint for our specific task.

**Navigate to the following branches to view the code and experimental pipelines for these models:**

* [`Qwen2.5-0.5B` Branch](https://github.com/Mesekir19/supervised_project/tree/Qwen2.5-0.5B)
* [`Qwen3-0.6B` Branch](https://github.com/Mesekir19/supervised_project/tree/Qwen3-0.6B)

## Setup

```bash
git clone https://github.com/Mesekir19/supervised_project.git
cd supervised_project
git checkout microsoft-phi-mini3.8
pip install -r requirements.txt

```

## Ethics & Sustainability

* All data is public OSM data; no PII collected.
* Total training/inference carbon footprint: ~20.49 g CO₂eq (tracked via
`codecarbon`).

## Authors

Peter Harmer, Mesekir Getachew Sahilu, Mulualem Asfaw Mekonnen,
Akob Tah Banjong, Usman Muhammad

Supervised by Prof. Christophe Cerisara, assisted by Gabriel Lauzzana.
MSc Natural Language Processing — Supervised Project, 2025–2026.

## References

* Manvi et al., [GeoLLM: Extracting Geospatial Knowledge from Large Language Models](https://arxiv.org/abs/2310.06213), 2024
* Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2022
* Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023
