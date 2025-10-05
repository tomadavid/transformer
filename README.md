# 🧠 Mini Transformer from Scratch  

### *There is no better way to understand something than building it yourself*
### (Otherwise is just a bunch of algebra and formulas that are assumed to work somehow)

This project implements a **Transformer-based language model** from scratch in PyTorch.  
It covers everything from **tokenization → dataset preparation → training loop → saving weights**.  

The main goal is **learning-by-building**: to understand how LLMs work at a lower level, rather than just using pre-built Hugging Face models.

The model was built after studying the transformer architecture and comes as a way to test my knowledge in practical way.

---

## ✨ Goals

- Build and train a **causal language model** (GPT-like) from first principles.  
- Explore **how embeddings, attention, and transformer blocks** work together.  
- Train on a **Wikipedia-derived dataset** to produce a model that can generate **Russian text**.  
- Save and reload both the model parameters and tokenizer for further experiments.

---

## ⚙️ Functional Decisions

- **Custom Tokenizer**  
  - Used Hugging Face’s `tokenizers` library with **Byte-Pair Encoding (BPE)**.  
  - Added a special `<EOS>` token to mark sentence endings.  
  - Chose **vocab size = 5000** to keep training manageable on limited hardware.  

- **Dataset**  
  - Used [Den4ikAI/russian_cleared_wikipedia](https://huggingface.co/datasets/Den4ikAI/russian_cleared_wikipedia).  
  - Each article was concatenated and split into fixed-length sequences for training.  
  - Implemented a custom `TextDataset` that outputs `(input_ids, target_ids)` pairs for next-token prediction.  

- **Model Architecture (`LLM` class)**  
  - A configurable Transformer decoder-only model (similar to GPT).  
  - Hyperparameters (e.g., `d_model`, `num_heads`, `num_layers`, `max_seq_len`) are stored in `params.py`.  
  - Final layer produces logits for each token in the vocabulary.  

- **Training**  
  - Optimizer: Adam with gradient clipping.  
  - Cross-entropy loss for next-token prediction.  
  - Progress bar prints batch progress per epoch.  
  - Model weights saved with:  
    ```python
    torch.save(model.state_dict(), "parameters.pt2")
    ```  

- **Saving & Reloading**  
  - Tokenizer saved as `bpe_tokenizer.json`.  
  - Model weights saved as `parameters.pt2`.  
  - Model must be rebuilt with same architecture before loading weights.

---

## 📂 Project Structure

```
.
├── dataset.py         # Custom TextDataset class
├── llm.py             # Transformer implementation (LLM class)
├── params.py          # Model hyperparameters
├── train.py           # Main training script (loads dataset, trains model)
├── bpe_tokenizer.json # Saved tokenizer (after training)
├── parameters.pt2     # Model weights (after training)
```

---

## 🔮 Next Steps / Ideas

- Evaluate model perplexity and sample text generation.
- Experiment with simple facts the model could have memorized.
- Explore attention score inside the layer and how is context captured
- Text if case termination of words is right (There are plenty in russian depending on the context)
- Explore and build ways to visualize the whole process from inside  

---

## 📌 Notes

This is not about building a state-of-the-art LLM, but rather about **understanding transformers end-to-end**.  
With ~1–10M parameters, the model won’t generate fluent Wikipedia-level text, but it will **learn patterns, word co-occurrences, and simple syntax**.  
