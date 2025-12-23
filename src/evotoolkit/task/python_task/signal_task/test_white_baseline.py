import torch
import numpy as np
import pickle
import os
import time
import torch.nn as nn
from whitebox_lib import WhiteBoxAttacker, DLR_Loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "RML2016.10a_dict.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "vtcnn2_100_-11_1213.pth") 
TEST_INDICES_PATH = os.path.join(BASE_DIR, "test_indices_high_snr.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128


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

def load_data():
    print(">>> Loading Data Dict...")
    with open(DATA_PATH, 'rb') as f: 
        Xd = pickle.load(f, encoding='latin1')
    
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    
    X_list = []
    lbl_list = []
    for mod in mods:
        for snr in snrs:
            X_list.append(Xd[(mod, snr)])
            for _ in range(Xd[(mod, snr)].shape[0]):
                lbl_list.append((mod, snr))
    X = np.vstack(X_list)
    lbl = np.array(lbl_list)
    
    print(">>> Normalizing Data (Global)...")
    dmin, dmax = np.min(X), np.max(X)
    X_norm = 2.0 * (X - dmin) / (dmax - dmin) - 1.0
    
    # 加载索引
    if not os.path.exists(TEST_INDICES_PATH): 
        raise FileNotFoundError(f"Indices file not found: {TEST_INDICES_PATH}")
        
    indices = np.load(TEST_INDICES_PATH)
    print(f">>> Loaded {len(indices)} indices from {TEST_INDICES_PATH}")
    
    # 提取测试数据
    X_eval = X_norm[indices]
    Y_eval = np.array([mods.index(lbl[x][0]) for x in indices])
    
    # 转 Tensor
    X_tensor = torch.tensor(X_eval, dtype=torch.float32).to(DEVICE)
    Y_tensor = torch.tensor(Y_eval, dtype=torch.long).to(DEVICE)
    
    return X_tensor, Y_tensor, len(mods)

def evaluate(model, x, y):

    model.eval()
    num_samples = x.shape[0]
    num_correct = 0
    
    with torch.no_grad():
        for i in range(0, num_samples, BATCH_SIZE):
            batch_x = x[i:i+BATCH_SIZE]
            batch_y = y[i:i+BATCH_SIZE]
            out = model(batch_x)
            pred = torch.argmax(out, dim=1)
            num_correct += (pred == batch_y).sum().item()
            
    return num_correct / num_samples

def run_attack_batch(attacker, X, Y, method, loss_type):
    """
    分批运行攻击，并实时计算准确率
    """
    num_samples = X.shape[0]
    adv_samples = []
    
    print(f"   > Attack method: {method}-{loss_type} | Samples: {num_samples}")
    
    start_time = time.time()
    
    # 分批处理
    for i in range(0, num_samples, BATCH_SIZE):
        batch_x = X[i:i+BATCH_SIZE]
        batch_y = Y[i:i+BATCH_SIZE]
        
        # 运行攻击
        batch_adv = attacker.batch_attack(batch_x, batch_y, method=method, loss_type=loss_type)
        adv_samples.append(batch_adv)
        
        if (i // BATCH_SIZE) % 10 == 0 and i > 0:
            print(f"     Processed {i}/{num_samples} samples...")
            
    total_time = time.time() - start_time
    
    # 合并结果
    X_adv_all = torch.cat(adv_samples, dim=0)
    
    return X_adv_all, total_time

if __name__ == "__main__":
    # 准备数据和模型
    X, Y, num_classes = load_data()
    # X, Y = X[:1000], Y[:1000] 
    
    model = VTCNN2(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 初始准确率
    print("\n>>> Verifying Clean Accuracy...")
    clean_acc = evaluate(model, X, Y)
    print(f"[Baseline] Clean Accuracy: {clean_acc*100:.2f}%")
        
    # 设置攻击参数
    EPSILON = 0.005  
    STEPS = 50       
    
    print(f"\n[Config] Epsilon: {EPSILON}, Steps: {STEPS}")
    
    attacker = WhiteBoxAttacker(model, epsilon=EPSILON, n_iter=STEPS, device=DEVICE)

    # (A) Standard PGD (CE Loss)
    print("\n[1/4] Running PGD (CE)...")
    adv_pgd, t_pgd = run_attack_batch(attacker, X, Y, 'pgd', 'ce')
    acc_pgd = evaluate(model, adv_pgd, Y)
    print(f"Result: {acc_pgd*100:.2f}% (Time: {t_pgd:.1f}s)")
    
    # (B) Standard PGD (DLR Loss)
    print("\n[2/4] Running PGD (DLR)...")
    adv_pgd_dlr, t_pgd_dlr = run_attack_batch(attacker, X, Y, 'pgd', 'dlr')
    acc_pgd_dlr = evaluate(model, adv_pgd_dlr, Y)
    print(f"Result: {acc_pgd_dlr*100:.2f}% (Time: {t_pgd_dlr:.1f}s)")

    # (C) APGD (DLR Loss)
    print("\n[3/4] Running APGD (DLR)...")
    adv_apgd, t_apgd = run_attack_batch(attacker, X, Y, 'apgd', 'dlr')
    acc_apgd = evaluate(model, adv_apgd, Y)
    print(f"Result: {acc_apgd*100:.2f}% (Time: {t_apgd:.1f}s)")
    
    # (D) APGD (CE Loss)
    print("\n[4/4] Running APGD (CE)...")
    adv_apgd_ce, t_apgd_ce = run_attack_batch(attacker, X, Y, 'apgd', 'ce')
    acc_apgd_ce = evaluate(model, adv_apgd_ce, Y)
    print(f"Result: {acc_apgd_ce*100:.2f}% (Time: {t_apgd_ce:.1f}s)")
