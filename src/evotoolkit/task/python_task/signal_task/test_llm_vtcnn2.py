import os
import pickle
import numpy as np
import json
import torch
import torch.nn as nn
import warnings
import evotoolkit
from evotoolkit.task.python_task import PythonTask, EvoEngineerPythonInterface, AdversarialAttackTask
from evotoolkit.tools.llm import HttpsApi
from evotoolkit.core import Solution, EvaluationResult
from signal_task import SignalAttackTask
import foolbox as fb

json.JSONEncoder.default = lambda self, obj: \
    obj.tolist() if torch.is_tensor(obj) else json.JSONEncoder.default(self, obj)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "RML2016.10a_dict.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "vtcnn2_100_-11_1213.pth") 
EVO_INDICES_PATH = os.path.join(BASE_DIR, "evolution_indices.npy")

class VTCNN2(nn.Module):
    def __init__(self, num_classes=11, dropout=0.5, input_shape=(1, 2, 128)):
        super().__init__()
        self.dropout_rate = dropout
        self.pad1 = nn.ZeroPad2d((2, 2, 0, 0))   
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.pad2 = nn.ZeroPad2d((2, 2, 0, 0))
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.pad1(dummy_input)
            x = self.relu1(self.conv1(x))
            x = self.dropout1(x)
            x = self.pad2(x)
            x = self.relu2(self.conv2(x))
            x = self.dropout2(x)
            conv_output_size = x.flatten(1).shape[1]
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 2: 
            x = x.unsqueeze(0)
        if x.dim() == 3:
            x = x.unsqueeze(1) 
        x = self.pad1(x)
        x = self.relu1(self.conv1(x))
        # x = self.dropout1(x) 
        x = self.pad2(x)
        x = self.relu2(self.conv2(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

print("1. Loading Data...")
with open(DATA_PATH, 'rb') as f:
    Xd = pickle.load(f, encoding='latin1')

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X, lbl = [], []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)
lbl = np.array(lbl)

print("2. Re-calculating Normalization Parameters (Seed 2016)...")
np.random.seed(2016)
train_idx = []
for snr in snrs:
    idx_snr = [i for i, x in enumerate(lbl) if int(x[1]) == snr]
    if not idx_snr: continue
    np.random.shuffle(idx_snr)
    n_train_snr = int(0.5 * len(idx_snr))
    train_idx.extend(idx_snr[:n_train_snr])

X_train_raw = X[train_idx]
data_min = np.min(X_train_raw)
data_max = np.max(X_train_raw)
print(f"   Train Min: {data_min}, Max: {data_max}")

print("3. Applying Normalization & Clipping...")
X_norm = 2.0 * (X - data_min) / (data_max - data_min) - 1.0
X_final = np.clip(X_norm, -1.0, 1.0)

print("4. Loading Model...")
num_classes = len(mods)
model = VTCNN2(num_classes=num_classes).to(device)

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Warning: Model path not found! Using random weights.")

model.eval()

fmodel = fb.PyTorchModel(model, bounds=(-1, 1), device=device)

print(f"5. Loading Evolution Indices from {EVO_INDICES_PATH}...")
if not os.path.exists(EVO_INDICES_PATH):
    raise FileNotFoundError("Evolution indices not found! Please run the sampling script first.")

evo_indices = np.load(EVO_INDICES_PATH)
X_evo_np = X_final[evo_indices]
Y_evo_np = np.array([mods.index(lbl[x][0]) for x in evo_indices])

X_eval = torch.tensor(X_evo_np, dtype=torch.float32).to(device)
Y_eval = torch.tensor(Y_evo_np, dtype=torch.long).to(device)

print(f"   Evolution Set Size: {len(X_eval)}")
print(f"   Data Range: [{X_eval.min():.4f}, {X_eval.max():.4f}]")

print(f"\n[Pre-check] Generating robust starting points for EVOLUTION...")
print("   Trying robust initialization (directions=1000, steps=1000)...")

init_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(directions=1000, steps=1000)

init_advs, _, is_adv = init_attack(fmodel, X_eval, criterion=fb.criteria.Misclassification(Y_eval), epsilons=None)

success_indices = [i for i, success in enumerate(is_adv) if success.item()]

print(f"   Init Failed    : {len(X_eval) - len(success_indices)} (Dropped)")
print(f"   Remaining Valid: {len(success_indices)}")

if len(success_indices) == 0:
    raise RuntimeError("All evolution samples failed initialization! Check model accuracy or data.")

X_eval = X_eval[success_indices]
Y_eval = Y_eval[success_indices]
starting_points = init_advs[success_indices]

signal_task = SignalAttackTask(model, X_eval, Y_eval, attack_steps=1000,starting_points=starting_points)

interface = EvoEngineerPythonInterface(signal_task)

llm_api = HttpsApi(
    api_url='api.bltcy.ai', 
    key='LLM_API_KEY',
    model='deepseek-v3',
    timeout=60
)

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f'./signal_attack_raw_{timestamp}'

print("STARTING EVOLUTION (Normalized Data Mode)")

result = evotoolkit.solve(
    interface=interface,
    output_path=output_dir,
    running_llm=llm_api,
    max_generations=100,
    pop_size=10,
    max_sample_nums=1000, 
    debug_mode=False
)

print("EVOLUTION COMPLETED!")

if result:
    print(f"\nBest Solution Found:")
    print("-" * 30)
    print(result.sol_string)
    print("-" * 30)
    
    if result.evaluation_res and result.evaluation_res.score is not None:
        final_dist = -result.evaluation_res.score
        print(f"\nBest Mean L2 Distance:{final_dist:.4f}")
        print(f"Result saves at： {output_dir}")
    else:
        print("\nResult valid but no score available.")
else:
    print("\nEvolution failed to produce a valid solution.")
