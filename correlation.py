import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

val_preds = np.load('val_preds.npy')
val_true = np.load('val_true.npy')

print(classification_report(val_true, val_preds, target_names=['0 Pers', '1 Pers', '2 Pers', '3 Pers']))

cm = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['0 Pers', '1 Pers', '2 Pers', '3 Pers'],
            yticklabels=['0 Pers', '1 Pers', '2 Pers', '3 Pers'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Done")