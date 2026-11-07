# Mechanistic-Interpretability-of-LLM-Refusal-Behaviour

> **For education and understanding only.** This repository shows how a small open model, Qwen3.5-0.8B, represents refusal inside its layers, so that people can see it, measure it and reason about it. It is not a tool for getting harmful output:
>
> - Replies to harmful prompts are cut to their first few words in the code, the saved results and the interactive page.
> - All edits are temporary hooks while the model runs. No modified weights are ever saved.
>
> Please do not use these methods to cause harm.

![The bomb vector, the pie vector, and the refusal direction between them](images/01_bomb_pie_refusal.png)

Ask the model **"how to make a bomb"** and **"how to make a pie"**. Inside the model, each question becomes a vector of 1,024 numbers. The two vectors are almost the same length (11.06 and 11.14) but point in slightly different directions. Their difference, **u**, is a *refusal direction*: when the model's vector leans along u, the model refuses. In every picture, red is the bomb side and blue is the pie side.

This repo finds u in Qwen3.5-0.8B and changes the behaviour in two ways:

- **Way 1: remove the direction.** Take u out of every layer's mixer write, MLP write and output.
  - The bomb prompt no longer gets "I cannot …"; it only gets a warning.
  - On 20 new harmful prompts, refusals fall from 18 to 12.
- **Way 2: flip 50 neurons.** Find the 50 MLP neurons, out of 86,016, that push hardest along u on harmful prompts, and multiply their activations by −1.
  - On the same 20 prompts, refusals fall from 18 to 4.
- **Harmless prompts are unaffected.** Both edits still answer all 10 new harmless prompts.

## Explore

![The interactive page](images/31_explore_page.png)

The interactive page is live at **[ashishkhar.github.io/Mechanistic-Interpretability-of-LLM-Refusal-Behaviour](https://ashishkhar.github.io/Mechanistic-Interpretability-of-LLM-Refusal-Behaviour/)**, with every measurement inside the file, so it works offline too. On the page you can:

- Pick an edit: the original model; way 1, removing 25–100% of u; or way 2, zeroing or flipping the top 10, 30 or 50 neurons.
- Pick the residual stream, the mixer write or the MLP write, measured along u or by length.
- Step through all 24 layers and every token.
- See bomb and pie as arrows, follow the value layer by layer, and see what the model says first.

## Run

```bash
git clone https://github.com/ashishkhar/Mechanistic-Interpretability-of-LLM-Refusal-Behaviour.git
cd Mechanistic-Interpretability-of-LLM-Refusal-Behaviour
uv run python refusal_qwen.py
```

- The first run downloads `Qwen/Qwen3.5-0.8B` (about 1.6 GB) from Hugging Face.
- It takes about 8 minutes on an Apple-silicon Mac, and longer on CPU. A CUDA GPU is used when available.
- It prints a summary and writes `results.json`, the numbers behind `index.html`.
- You need Python 3.12 and [uv](https://docs.astral.sh/uv/). Without uv, install `torch`, `transformers` and `accelerate` and run the script with `python`.

| option | what it does |
|---|---|
| `--model PATH` | use a local copy of the model instead of downloading it |
| `--way 1` / `--way 2` | run only one of the two ways |
| `--n 50` | prompts per side for the averages (default 100) |
| `--device cpu` | force a device (default: cuda, then mps, then cpu) |
| `--quick` | one-minute smoke test that writes `results_quick.json` |

## How it works

### 1. Measuring a vector against u

![Length, cosine and signed projection](images/04_three_measurements.png)

A vector has a length and a direction. Against u, it can be measured three ways:

- **Length.** Bomb and pie are almost the same: 11.06 against 11.14.
- **Cosine.** +0.33 against −0.35.
- **Signed projection**, the vector's shadow on u. +3.64 against −3.87.

So the two prompts differ in direction, not size.

### 2. Inside the model

![Tokens pass through 24 layers into the scores for the next word](images/07_model_24_layers.png)

Qwen3.5-0.8B has 24 layers: 18 Gated DeltaNet and 6 full attention. Each layer reads a running total of 1,024 numbers per token, the *residual stream*, and adds a mixer write and an MLP write to it. The first word of the answer is chosen at the last token.

![Every layer and every token for bomb, pie and their difference](images/10_bomb_pie_heatmaps.png)

Each cell is one vector measured along u. Up to the word "bomb" or "pie" the prompts are identical. After it, the bomb prompt turns red and the pie prompt turns blue, mostly in the top layers.

![Where the bomb − pie gap is built](images/11_gap_by_layer.png)

At the last token, layers 20–23 write 5.08 of the 7.52 gap. The biggest single write is the layer-23 MLP.

### 3. From one pair to 100 prompts

![Means of 100 harmful and 100 harmless prompts](images/13_means_heatmaps.png)

Every chat prompt ends with the same 9 post-instruction tokens, so many prompts can be lined up and averaged. Averaging 100 harmful and 100 harmless prompts shows the same direction separating harmful requests from harmless ones.

### 4. Way 1: remove u at every layer

![Where u was removed and what the replies did](images/17_removal_attempts.png)

Each vector x becomes x − (x · u) u, which takes away its shadow on u. Removing u from three cells of the layer-23 MLP, or from the top 10 cells in layers 0–19, still leaves the refusal. Removing it from whole layers takes away "I cannot".

![Removing u early: the top layers write it back](images/19_written_back.png)

When u is removed only in layers 0–19, layers 20–23 write part of it back, up to 0.85 by the last layer. Removing it at every layer keeps it at 0.

![Position along u goes to zero while the length barely changes](images/20_direction_not_length.png)

With u removed everywhere, the position along u drops from +3.64 to 0, while the length only goes from 11.06 to 11.36. Refusal lives in the direction.

### 5. Way 2: flip 50 neurons

![The 50 neurons that push hardest toward refusal](images/25_top50_neurons.png)

Each MLP neuron gets a score: (its average activation on harmful prompts − on harmless prompts) × (its output arrow · u). The top neuron is neuron 217 in layer 23. The top 50 cover 38% of the MLPs' total push, and most of them sit in the top layers.

![Zeroing or flipping the top 10, 30 and 50 neurons](images/28_neuron_results.png)

Zeroing or flipping 10 neurons is not enough. Flipping all 50 makes both harmful test prompts answer, while pie and 17 + 28 are answered throughout.

### 6. On new prompts

![Refusals on 20 new harmful and 10 new harmless prompts](images/29_heldout_check.png)

![Summary of both ways](images/30_summary.png)

## Files

| path | what it is |
|---|---|
| `refusal_qwen.py` | standalone script: finds u, runs both ways, saves the results |
| `index.html` | the interactive page |
| `prompts.json` | the prompt sample |
| `results.json` | every number shown on the page |
| `images/` | the figures in this README |

## Credits

- **Model:** Qwen3.5-0.8B by the Qwen team, Apache-2.0.
- **Harmful prompts:** public red-teaming sets, named in each entry of `prompts.json`: AdvBench, HarmBench text behaviours, TDC 2023 and an adversarial-training set. Chemical, biological, copyright and contextual items were removed.
- **Harmless prompts:** Alpaca-style instructions.
- **Method:** the difference-of-means refusal direction and its removal at every layer follow Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024).
