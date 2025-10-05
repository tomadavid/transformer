# Model Parameters (mini transformer)
vocab_size=5000
d_model=128
num_layers=2
num_heads=2
d_ff=512
d_k = d_model//num_heads
d_v = d_model//num_heads
max_seq_len=128
dropout=0.1

# Training Parameters
batch_size=4
epochs=1
