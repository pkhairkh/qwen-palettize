# Research Papers

20 papers relevant to the qwen-palettize enhancement roadmap. Total: 44MB.

## Gumbel-Softmax & STE (our approach)

| Paper | File | Relevance |
|-------|------|-----------|
| Gumbel-Softmax (Jang et al. 2017) | `1611.01144_Gumbel-Softmax_Jang2017.pdf` | Core: our indices training method |
| Concrete Distribution (Maddison et al. 2017) | `1611.00712_ConcreteDistribution_Maddison2017.pdf` | Core: alternative relaxation (Gumbel-Softmax sibling) |
| STE (Bengio et al. 2013) | `1308.3432_STE_Bengio2013.pdf` | Core: Straight-Through Estimator (our backward) |
| Discrete VAE (Rolfe 2016) | `1609.02200_DiscreteVAE_Rolfe2016.pdf` | Related: discrete latent variables |
| BNN (Courbariaux 2016) | `1602.02830_BNN_Courbariaux2016.pdf` | Related: binary quantization + tight logit clamp |
| **LLT (Wang et al. CVPR 2022)** | `LLT_Wang_CVPR2022.pdf` | **Key: temperatured softmax + STE + 1/√Nᵢ gradient rescaling** |

## Optimizers

| Paper | File | Relevance |
|-------|------|-----------|
| Adam (Kingma & Ba 2015) | `1412.6980_Adam_Kingma2015.pdf` | Core: our optimizer base |
| AdamW (Loshchilov & Hutter 2019) | `1711.05101_AdamW_Loshchilov2019.pdf` | Core: our optimizer (fp32 master) |
| Dissecting Adam (Balles & Hennig 2017) | `1705.07774_DissectingAdam_Balles2017.pdf` | Analysis: Adam sign/magnitude |
| Shampoo (Bernstein 2024) | `2409.20325_Shampoo_Bernstein2024.pdf` | Related: Muon lineage |

## Quantization (referenced in enhancement patches)

| Paper | File | Relevance |
|-------|------|-----------|
| LUT-Q (Cardinaux et al. 2018) | `1811.05355_LUTQ_Cardinaux2018.pdf` | Related: LUT quantization with k-means + STE |
| QAT Oscillations (Nagel et al. 2022) | `2203.11086_QAT_Oscillations_Nagel2022.pdf` | Patch 7: freeze_settled_palettes (Nagel fix) |
| GPTQ (Frantar et al. 2023) | `2210.17323_GPTQ_Frantar2023.pdf` | Comparison: closed-form calibration (rejected) |
| QLoRA (Dettmers et al. 2023) | `2305.14314_QLoRA_Dettmers2023.pdf` | Comparison: NF4 + LoRA (rejected) |
| AWQ (Lin et al. 2024) | `2306.00978_AWQ_Lin2024.pdf` | Comparison: activation-aware scaling (rejected) |
| SqueezeLLM (Kim et al. 2024) | `2306.07629_SqueezeLLM_Kim2024.pdf` | Comparison: dense/sparse split (rejected) |
| AQLM (Egiazarian et al. 2024) | `2401.06118_AQLM_Egiazarian2024.pdf` | Comparison: additive VQ (rejected) |

## Other (from kernel/architecture research)

| Paper | File | Relevance |
|-------|------|-----------|
| ref1 | `1902.08153_ref1.pdf` | Kernel research reference |
| ref2 | `2306.16817_ref2.pdf` | Kernel research reference |
| ref3 | `2310.08659_ref3.pdf` | Kernel research reference |
