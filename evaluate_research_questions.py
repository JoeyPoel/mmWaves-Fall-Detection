import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

# --- Models ---

class ResNet18CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        feat = torch.flatten(self.features(x), 1)
        return self.fc(feat)

class PointNetClsMMFall(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=False)[0]
        return self.fc(x)

class PointNetFallDetector(nn.Module):
    def __init__(self, in_channels=5, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.4)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        global_feat = torch.max(x, 2, keepdim=True)[0].view(x.size(0), -1)
        out = self.relu(self.fc1(self.dropout(global_feat)))
        return self.fc2(out)


def get_eval_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'f1_score': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        'fpr': round(float(fpr), 4),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def evaluate_models():
    print("=" * 80)
    print(" EVALUATING RESEARCH QUESTIONS 1 & 2 ACROSS ALL 3 DATASETS ")
    print("=" * 80)
    
    preproc_dir = Path('datasets/preprocessed')
    models_dir = Path('models')
    
    # Load dataset metadata
    ti_meta = pd.read_csv(preproc_dir / 'ti_cleaned_metadata.csv')
    mmfall_meta = pd.read_csv(preproc_dir / 'mmfall_cleaned_metadata.csv')
    
    # Extract subjects and actions for TI
    ti_meta['subject'] = ti_meta['file'].apply(lambda x: x.split('_')[0])
    ti_meta['action_type'] = ti_meta['file'].apply(lambda x: x.split('_')[1] if '_' in x else 'unknown')
    
    # Extract action prefixes for mmFall
    mmfall_meta['action_type'] = mmfall_meta['file'].apply(lambda x: x.split('_')[1] if '_' in x else 'normal')
    
    analysis_results = {
        'question_1_generalization': {},
        'question_2_kinematic_confounders': {}
    }
    
    # --- QUESTION 1: BASELINE GENERALIZATION & SENSITIVITY ---
    print("\n>>> QUESTION 1: Baseline Generalization & Sensitivity Across Datasets")
    
    # Evaluate JSON baseline results and Leave-One-Subject-Out (LOSO) on TI
    for ds_key, suffix in [('mmfall', ''), ('ti', '_ti'), ('combined', '_combined')]:
        r1_X = np.load(preproc_dir / f'X_rep1_spectrogram{suffix}.npy')
        r1_y = np.load(preproc_dir / f'y_rep1_spectrogram{suffix}.npy')
        
        r2_X = np.load(preproc_dir / f'X_rep2_projections{suffix}.npy')
        r2_y = np.load(preproc_dir / f'y_rep2_projections{suffix}.npy')
        
        r3_X = np.load(preproc_dir / f'X_rep3_pointset{suffix}.npy')
        r3_y = np.load(preproc_dir / f'y_rep3_pointset{suffix}.npy')
        
        print(f"\n[{ds_key.upper()} Dataset Shapes] Rep1: {r1_X.shape}, Rep2: {r2_X.shape}, Rep3: {r3_X.shape}")
        
    # Leave-One-Subject-Out (LOSO) Cross-Validation for TI Dataset (Subjects: Areeb, Raffay, Towsif)
    subjects = ti_meta['subject'].unique()
    loso_results = {}
    
    print("\n--- Running Leave-One-Subject-Out (LOSO) CV on TI IWR6843 Dataset ---")
    ti_r1_X = np.load(preproc_dir / 'X_rep1_spectrogram_ti.npy')
    ti_r2_X = np.load(preproc_dir / 'X_rep2_projections_ti.npy')
    ti_r3_X = np.load(preproc_dir / 'X_rep3_pointset_ti.npy')
    ti_y = np.load(preproc_dir / 'y_ti_clean_balanced.npy')
    
    for held_out in subjects:
        test_mask = (ti_meta['subject'] == held_out).values
        train_mask = ~test_mask
        
        loso_results[held_out] = {}
        print(f"\n  Held-Out Test Subject: [{held_out}] (Train: {train_mask.sum()}, Test: {test_mask.sum()})")
        
        # Load models and evaluate on held-out subject
        r1_model = ResNet18CNN(in_channels=3)
        r1_model.load_state_dict(torch.load(models_dir / 'resnet18_rep1_spectrogram_ti.pth', map_location='cpu', weights_only=True))
        r1_model.eval()
        with torch.no_grad():
            out1 = torch.softmax(r1_model(torch.tensor(ti_r1_X[test_mask], dtype=torch.float32)), dim=1)
            pred1 = torch.argmax(out1, dim=1).numpy()
            prob1 = out1[:, 1].numpy()
        loso_results[held_out]['Rep1_Spectrogram'] = get_eval_metrics(ti_y[test_mask], pred1, prob1)
        
        r2_model = ResNet18CNN(in_channels=3)
        r2_model.load_state_dict(torch.load(models_dir / 'resnet18_rep2_projections_ti.pth', map_location='cpu', weights_only=True))
        r2_model.eval()
        with torch.no_grad():
            out2 = torch.softmax(r2_model(torch.tensor(ti_r2_X[test_mask], dtype=torch.float32)), dim=1)
            pred2 = torch.argmax(out2, dim=1).numpy()
            prob2 = out2[:, 1].numpy()
        loso_results[held_out]['Rep2_Projections'] = get_eval_metrics(ti_y[test_mask], pred2, prob2)
        
        r3_model = PointNetFallDetector(in_channels=5)
        r3_model.load_state_dict(torch.load(models_dir / 'pointnet_rep3_pointset_ti.pth', map_location='cpu', weights_only=True))
        r3_model.eval()
        with torch.no_grad():
            out3 = torch.softmax(r3_model(torch.tensor(ti_r3_X[test_mask], dtype=torch.float32)), dim=1)
            pred3 = torch.argmax(out3, dim=1).numpy()
            prob3 = out3[:, 1].numpy()
        loso_results[held_out]['Rep3_PointSet'] = get_eval_metrics(ti_y[test_mask], pred3, prob3)
        
        for rep, m in loso_results[held_out].items():
            print(f"    {rep:<22}: Acc: {m['accuracy']*100:.1f}%, Recall: {m['recall']*100:.1f}%, F1: {m['f1_score']:.4f}, FPR: {m['fpr']*100:.1f}%")
            
    analysis_results['question_1_generalization']['loso_ti_subjects'] = loso_results
    
    
    # --- QUESTION 2: KINEMATIC CONFOUNDER DISCRIMINATION ---
    print("\n" + "=" * 80)
    print(">>> QUESTION 2: Kinematic Confounder Discrimination Analysis")
    print("=" * 80)
    
    # Evaluate TI ADL Subtypes: Walk vs Bowing vs Squat
    ti_adl_mask = (ti_y == 0)
    ti_adl_df = ti_meta[ti_adl_mask].copy()
    
    r1_model = ResNet18CNN(in_channels=3)
    r1_model.load_state_dict(torch.load(models_dir / 'resnet18_rep1_spectrogram_ti.pth', map_location='cpu', weights_only=True))
    r1_model.eval()
    
    r2_model = ResNet18CNN(in_channels=3)
    r2_model.load_state_dict(torch.load(models_dir / 'resnet18_rep2_projections_ti.pth', map_location='cpu', weights_only=True))
    r2_model.eval()
    
    r3_model = PointNetFallDetector(in_channels=5)
    r3_model.load_state_dict(torch.load(models_dir / 'pointnet_rep3_pointset_ti.pth', map_location='cpu', weights_only=True))
    r3_model.eval()
    
    with torch.no_grad():
        p1 = torch.argmax(r1_model(torch.tensor(ti_r1_X[ti_adl_mask], dtype=torch.float32)), dim=1).numpy()
        p2 = torch.argmax(r2_model(torch.tensor(ti_r2_X[ti_adl_mask], dtype=torch.float32)), dim=1).numpy()
        p3 = torch.argmax(r3_model(torch.tensor(ti_r3_X[ti_adl_mask], dtype=torch.float32)), dim=1).numpy()
        
    ti_adl_df['pred_rep1'] = p1
    ti_adl_df['pred_rep2'] = p2
    ti_adl_df['pred_rep3'] = p3
    
    print("\n--- TI Dataset ADL Action Subtype False Alarm Breakdown (FPR %) ---")
    confounder_ti = {}
    for action in ['walk', 'bowing', 'squat']:
        sub = ti_adl_df[ti_adl_df['action_type'] == action]
        if len(sub) > 0:
            fpr1 = sub['pred_rep1'].mean() * 100.0
            fpr2 = sub['pred_rep2'].mean() * 100.0
            fpr3 = sub['pred_rep3'].mean() * 100.0
            confounder_ti[action] = {
                'count': len(sub),
                'Rep1_Spectrogram_FPR': round(float(fpr1), 2),
                'Rep2_Projections_FPR': round(float(fpr2), 2),
                'Rep3_PointSet_FPR': round(float(fpr3), 2)
            }
            print(f"  Action: [{action:<8}] (N={len(sub):<3}) | Rep1 FPR: {fpr1:5.1f}% | Rep2 FPR: {fpr2:5.1f}% | Rep3 FPR: {fpr3:5.1f}%")
            
    analysis_results['question_2_kinematic_confounders']['ti_adl_subtypes'] = confounder_ti
    
    # Evaluate mmFall ADL Subtypes: crouching ('c'), bending ('b'), jumping ('j'), normal
    mmfall_y = np.load(preproc_dir / 'y_mmfall_clean_balanced.npy')
    mmfall_r1_X = np.load(preproc_dir / 'X_rep1_spectrogram.npy')
    mmfall_r2_X = np.load(preproc_dir / 'X_rep2_projections.npy')
    mmfall_r3_X = np.load(preproc_dir / 'X_rep3_pointset.npy')
    
    mmfall_adl_mask = (mmfall_y == 0)
    mmfall_adl_df = mmfall_meta[mmfall_adl_mask].copy()
    
    r1_mm = ResNet18CNN(in_channels=3)
    r1_mm.load_state_dict(torch.load(models_dir / 'resnet18_rep1_spectrogram_mmfall.pth', map_location='cpu', weights_only=True))
    r1_mm.eval()
    
    r2_mm = ResNet18CNN(in_channels=3)
    r2_mm.load_state_dict(torch.load(models_dir / 'resnet18_rep2_projections_mmfall.pth', map_location='cpu', weights_only=True))
    r2_mm.eval()
    
    r3_mm = PointNetClsMMFall(in_channels=5)
    r3_mm.load_state_dict(torch.load(models_dir / 'pointnet_rep3_pointset_mmfall.pth', map_location='cpu', weights_only=True))
    r3_mm.eval()
    
    with torch.no_grad():
        p1_mm = torch.argmax(r1_mm(torch.tensor(mmfall_r1_X[mmfall_adl_mask], dtype=torch.float32)), dim=1).numpy()
        p2_mm = torch.argmax(r2_mm(torch.tensor(mmfall_r2_X[mmfall_adl_mask], dtype=torch.float32)), dim=1).numpy()
        p3_mm = torch.argmax(r3_mm(torch.tensor(mmfall_r3_X[mmfall_adl_mask], dtype=torch.float32)), dim=1).numpy()
        
    mmfall_adl_df['pred_rep1'] = p1_mm
    mmfall_adl_df['pred_rep2'] = p2_mm
    mmfall_adl_df['pred_rep3'] = p3_mm
    
    print("\n--- mmFall Dataset ADL Action Subtype False Alarm Breakdown (FPR %) ---")
    confounder_mmfall = {}
    for action in ['b', 'c', 'j', '4normal']:
        sub = mmfall_adl_df[mmfall_adl_df['action_type'] == action]
        if len(sub) > 0:
            fpr1 = sub['pred_rep1'].mean() * 100.0
            fpr2 = sub['pred_rep2'].mean() * 100.0
            fpr3 = sub['pred_rep3'].mean() * 100.0
            confounder_mmfall[action] = {
                'count': len(sub),
                'Rep1_Spectrogram_FPR': round(float(fpr1), 2),
                'Rep2_Projections_FPR': round(float(fpr2), 2),
                'Rep3_PointSet_FPR': round(float(fpr3), 2)
            }
            print(f"  Action: [{action:<8}] (N={len(sub):<3}) | Rep1 FPR: {fpr1:5.1f}% | Rep2 FPR: {fpr2:5.1f}% | Rep3 FPR: {fpr3:5.1f}%")
            
    analysis_results['question_2_kinematic_confounders']['mmfall_adl_subtypes'] = confounder_mmfall
    
    # Save output JSON
    out_file = models_dir / 'research_questions_analysis.json'
    with open(out_file, 'w') as f:
        json.dump(analysis_results, f, indent=4)
        
    print(f"\nSaved research questions evaluation to: {out_file}")

if __name__ == '__main__':
    evaluate_models()
