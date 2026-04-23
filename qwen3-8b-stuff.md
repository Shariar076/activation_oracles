## Qwen3-8B Experiments & Findings

### Training Setup
- Trained on a **single H100 GPU**, taking approximately **10 H100-hours**
- Uses LoRA adapters (rank 64, alpha 128, dropout 0.05, all linear layers)
- Activations collected from layers at **25%, 50%, and 75% depth** during training; **50% depth** used at evaluation

---

### Secret-Keeping Evaluations (Figure 19)

| Training Config | Taboo | Gender | SSC |
|---|---|---|---|
| Original (untrained) | ~0% | 50% (chance) | — |
| SPQA Only | 4.4% | 90.5% | — |
| SPQA + Classification | — | 93.0% | — |
| Full Dataset | **20.5%** | **99.5%** | — |

- On **Taboo**, the SPQA-only model performs near zero, and benefits *substantially* from adding classification and context-prediction data — a contrast to Gemma-2-9B-IT where SPQA-only was already competitive
- This suggests the **value of training diversity is model-dependent**

---

### PersonaQA (Figure 16 & 18)

| Training Config | Open-Ended Accuracy | Binary Yes/No Accuracy |
|---|---|---|
| Original model | ~4.5% | ~49.7% (chance) |
| SPQA Only | ~7.8% | 58.1% |
| SPQA + Classification | — | — |
| Full Dataset | **12.2%** | **69.6%** |

- Scaling the training mixture yields **large gains** on Qwen3-8B for PersonaQA — notably larger than observed on other models
- The AO **outperforms the fine-tuned PersonaQA model itself** on binary yes/no questions (69.6% vs ~54.2%), a result described as "surprising" and replicated across hyperparameter configurations

---

### Emergent Misalignment (Figure 10)

- Qwen3-8B fine-tuned on **risky financial advice** was the *only* model across all six fine-tuned variants to exhibit significant emergent misalignment: **18% ± 8%**
- All other Qwen3-8B fine-tuning conditions (extreme sports, medical) showed **~0–1%** misalignment
- The inclusion of 50% UltraChat chat data during fine-tuning largely suppressed broad misalignment

---

### Training Diversity Ablation (Figure 7)

Three configurations tested on Qwen3-8B specifically:

| Config | Classification | PersonaQA | Taboo |
|---|---|---|---|
| SPQA + Classification (400k) | 68.5% | 7.8% | 30.2% |
| Truncated Full Mixture (400k) | 68.6% | 10.8% | 45.9% |
| Full Dataset (1M) | **70.0%** | **12.2%** | **49.5%** |

- The truncated full mixture **outperforms the data-matched baseline** despite allocating 60% of its budget to self-supervised context prediction (reducing supervised QA data) — indicating diversity helps even at fixed sample count
- Scaling to 1M examples provides further gains on top, suggesting both factors matter independently

---

### SAE Experiments (Appendix G)

- BatchTopK SAEs were trained on Qwen3-8B at 25%, 50%, and 75% depth layers, each with **65,000 features** and average L₀ ≈ 80
- Training on SAE-derived datasets produced **mixed results**: modest gains on some held-out classification tasks, but **minimal or negative impact** on OOD auditing benchmarks
- The AO did improve at verbalizing its own SAE features but did **not surpass the GPT-5-mini AutoInterp baseline**
- Given limited downstream benefit and nontrivial cost (~$1,000 in API credits for explanations alone), this direction was not scaled further

---

### Critical Notes

A few methodological points worth flagging for Qwen3-8B specifically:

- **PersonaQA accuracy is low** (~12% open-ended), and the paper acknowledges the skyline is unclear — linear probes on a related dataset only achieved 25%, so it's difficult to know how much headroom exists
- The PersonaQA fine-tuned models themselves **drop to near-chance on yes/no questions** (~54%), suggesting the biographical knowledge may not be robustly encoded in activations — which complicates interpretation of AO performance on this task
- The **larger gains from training diversity on Qwen3-8B** compared to Gemma suggest findings may not generalize uniformly across architectures, and model-specific tuning may remain necessary