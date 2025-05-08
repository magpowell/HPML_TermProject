import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("quantization_tests.csv")

# Correct the None qdtypes to torch.float32 (unquantized data type)
df['qdtype'] = df['qdtype'].fillna('torch.float32').astype(str)
df['qdtype'] = df['qdtype'].astype(str)

plt.style.use("ggplot")

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Plot 1: Average Accuracy
axs[0].bar(df['qdtype'], df['avg_rmse'], color='skyblue')
axs[0].set_ylabel("Average RMSE")
axs[0].set_title("Quantization Impact on Accuracy, Time, and Size")

# Plot 2: Average Time
axs[1].bar(df['qdtype'], 1000*df['avg_time'], color='salmon')
axs[1].set_ylabel("Average Inference Time (ms)")

# Plot 3: Model Size Reduction
axs[2].bar(df['qdtype'], df['quantized_model_size'], color='seagreen')
axs[2].set_ylabel("Quantized Model Size (%)")
axs[2].set_xlabel("Quantization Data Type")

plt.tight_layout()
plt.savefig("quantization_results.png")
