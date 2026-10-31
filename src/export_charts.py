import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(root_dir, 'assets')
models_dir = os.path.join(root_dir, 'models')

with open(os.path.join(models_dir, 'metrics.json'), 'r') as f:
    metrics = json.load(f)

with open(os.path.join(models_dir, 'history.json'), 'r') as f:
    history = json.load(f)

plt.style.use('dark_background')

# 1. Confusion Matrix
plt.figure(figsize=(6, 5), facecolor='#0d1117')
cm = np.array(metrics['confusion_matrix'])
acc = metrics['accuracy'] * 100
f1 = metrics['f1_score'] * 100

sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['Legitimate (0)', 'Fraud (1)'],
            yticklabels=['Legitimate (0)', 'Fraud (1)'],
            annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
plt.title(f"Confusion Matrix\nAccuracy: {acc:.2f}% | F1-Score: {f1:.2f}%", color='#00f2fe', fontsize=12, pad=12)
plt.ylabel('Actual Class', color='white', fontsize=11)
plt.xlabel('Predicted Class', color='white', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'confusion_matrix.png'), dpi=200, facecolor='#0d1117')
plt.close()

# 2. ROC Curve
plt.figure(figsize=(6, 5), facecolor='#0d1117')
fpr = metrics['curves']['fpr']
tpr = metrics['curves']['tpr']
plt.plot(fpr, tpr, color='#00f2fe', lw=2.5, label=f"ANN (AUC = {metrics['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], color='#8b5cf6', linestyle='--', lw=1.5, label='Random Guessing')
plt.title('Receiver Operating Characteristic (ROC)', color='#00f2fe', fontsize=12, pad=12)
plt.xlabel('False Positive Rate', color='white')
plt.ylabel('True Positive Rate', color='white')
plt.legend(loc='lower right', facecolor='#161b22', edgecolor='#30363d')
plt.grid(True, color='#21262d', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'roc_curve.png'), dpi=200, facecolor='#0d1117')
plt.close()

# 3. Precision-Recall Curve
plt.figure(figsize=(6, 5), facecolor='#0d1117')
prec_curve = metrics['curves']['precision_curve']
rec_curve = metrics['curves']['recall_curve']
plt.plot(rec_curve, prec_curve, color='#a855f7', lw=2.5, label=f"ANN (PR-AUC = {metrics['pr_auc']:.4f})")
plt.title('Precision-Recall Curve', color='#00f2fe', fontsize=12, pad=12)
plt.xlabel('Recall', color='white')
plt.ylabel('Precision', color='white')
plt.legend(loc='lower left', facecolor='#161b22', edgecolor='#30363d')
plt.grid(True, color='#21262d', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'pr_curve.png'), dpi=200, facecolor='#0d1117')
plt.close()

# 4. Training History (Loss & Accuracy)
epochs = range(1, len(history['loss']) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor='#0d1117')

ax1.set_facecolor('#0d1117')
ax1.plot(epochs, history['loss'], color='#00f2fe', lw=2, label='Train Loss')
ax1.plot(epochs, history['val_loss'], color='#f43f5e', lw=2, linestyle='--', label='Val Loss')
ax1.set_title('Binary Cross-Entropy Loss', color='#00f2fe', fontsize=11)
ax1.set_xlabel('Epochs', color='white')
ax1.set_ylabel('Loss', color='white')
ax1.legend(facecolor='#161b22', edgecolor='#30363d')
ax1.grid(True, color='#21262d', linestyle=':', alpha=0.6)

ax2.set_facecolor('#0d1117')
ax2.plot(epochs, history['accuracy'], color='#10b981', lw=2, label='Train Accuracy')
ax2.plot(epochs, history['val_accuracy'], color='#a855f7', lw=2, linestyle='--', label='Val Accuracy')
ax2.set_title('Model Accuracy', color='#00f2fe', fontsize=11)
ax2.set_xlabel('Epochs', color='white')
ax2.set_ylabel('Accuracy', color='white')
ax2.legend(facecolor='#161b22', edgecolor='#30363d')
ax2.grid(True, color='#21262d', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(assets_dir, 'training_history.png'), dpi=200, facecolor='#0d1117')
plt.close()

print('Successfully exported all 4 evaluation charts to assets/!')
