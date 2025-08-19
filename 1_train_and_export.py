import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import random
import os

# 1. Initialize H2O
# It's good practice to ensure H2O is initialized only once per session
# or to shut it down and re-initialize if running multiple times.
# Using h2o.init() without parameters often works fine,
# or you can specify ip and port: h2o.init(ip="127.0.0.1", port=54321)
h2o.init()

print("H2O Flow started.")

# --- Helper function to generate a simple DGA-like or legitimate domain ---
def generate_domain(is_dga, length_min=10, length_max=25):
    """Generates a synthetic domain name."""
    length = random.randint(length_min, length_max)
    if is_dga:
        # DGA-like: random mix of chars and digits
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        domain = ''.join(random.choice(chars) for _ in range(length))
        tlds = ['.com', '.net', '.org', '.info']
        return domain + random.choice(tlds)
    else:
        # Legitimate-like: more common words
        common_words = ["google", "microsoft", "amazon", "facebook", "apple",
                        "wikipedia", "youtube", "twitter", "reddit", "netflix"]
        base = random.choice(common_words)
        if len(base) < length:
            # Add some numbers or simple words if too short
            if random.random() < 0.5:
                base += str(random.randint(10, 99))
            else:
                base += random.choice(["news", "tech", "site", "blog"])
        tlds = ['.com', '.org', '.net']
        return base[:length] + random.choice(tlds) # Truncate if base is too long for chosen length

def calculate_entropy(s):
    """Calculates the Shannon entropy of a string."""
    from collections import Counter
    import math
    if not s:
        return 0
    probabilities = [float(c) / len(s) for c in Counter(s).values()]
    entropy = -sum([p * math.log(p) / math.log(2.0) for p in probabilities])
    return entropy

# --- Generate Synthetic Dataset ---
print("Generating synthetic dataset dga_dataset_train.csv...")
data = []
# Generate 1000 DGA domains and 1000 legitimate domains
for _ in range(1000):
    dga_domain = generate_domain(True)
    data.append([dga_domain, 1]) # 1 for DGA
for _ in range(1000):
    legit_domain = generate_domain(False)
    data.append([legit_domain, 0]) # 0 for Legitimate

df = pd.DataFrame(data, columns=['domain', 'label'])

# Feature Engineering for the dataset
df['length'] = df['domain'].apply(len)
df['entropy'] = df['domain'].apply(calculate_entropy)

# Save the dataset to CSV
output_csv_path = 'dga_dataset_train.csv'
df.to_csv(output_csv_path, index=False)
print(f"Dataset saved to {output_csv_path}")

# 2. Load the dataset into H2O Frame
hf = h2o.H2OFrame(df)

# Define predictors (x) and response (y)
x = ['length', 'entropy']
y = 'label'

# Ensure 'label' is treated as a categorical (factor) column for classification
hf[y] = hf[y].asfactor()

# 3. Run H2O AutoML
print("Starting H2O AutoML training...")
# Set a seed for reproducibility
aml = H2OAutoML(
    max_models=10,        # Limit the number of models to speed up training for the example
    seed=42,              # For reproducibility
    sort_metric='AUC',    # Metric to optimize for
    # Optional: exclude certain algorithms if you want to focus on specific ones
    # exclude_algos=['GLM', 'DRF', 'GBM', 'XGBoost', 'DeepLearning', 'StackedEnsemble']
)

aml.train(x=x, y=y, training_frame=hf)

print("AutoML training complete.")

# 4. Get the leader model
leader = aml.leader
print("\nLeaderboard:")
print(aml.leaderboard.as_data_frame())

print(f"\nLeader model ID: {leader.model_id}")

# 5. Export the leader model as a MOJO file
mojo_path = './model/DGA_Leader.zip'
if not os.path.exists('./model'):
    os.makedirs('./model') # Ensure the model directory exists

model_path = h2o.save_model(model=leader, path='./model/', force=True)
print(f"Leader model MOJO exported to: {model_path}")

# Rename the exported MOJO file to DGA_Leader.zip
if os.path.exists(model_path):
    # h2o.save_model returns the full path, which might include the model ID
    # We want to ensure it's named DGA_Leader.zip exactly.
    final_mojo_path = os.path.join('./model', 'DGA_Leader.zip')
    os.rename(model_path, final_mojo_path)
    print(f"Renamed MOJO to: {final_mojo_path}")
else:
    print("Error: MOJO file not found at expected path after save. Manual rename might be needed.")

# 6. Shut down H2O instance
h2o.cluster().shutdown(prompt=False)
print("H2O cluster shut down.")
