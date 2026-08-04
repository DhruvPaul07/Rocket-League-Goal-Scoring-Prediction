"""
Rocket League Goal-Scoring Prediction
Research Question: Can we accurately predict whether a team will score
within the next 10 seconds of game time?

Models: Decision Tree  vs  Logistic Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
                              classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH    = '/Users/dhruvpaul/Desktop/Dhruv/train_0.csv'
SAVE_DIR     = '/Users/dhruvpaul/Desktop/Dhruv/'
SAMPLE_SIZE  = 150_000
RANDOM_STATE = 42
TARGET       = 'team_A_scoring_within_10sec'
# Approximate Rocket League goal-line Y coordinates
GOAL_A_Y, GOAL_B_Y = 102.5, -102.5

plt.style.use('seaborn-v0_8-darkgrid')
BLUE, ORANGE = '#5B9BD5', '#ED7D31'


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & STRATIFIED SAMPLE
#     Technique: stratified sampling preserves the ~5.8% positive class ratio
# ═════════════════════════════════════════════════════════════════════════════
print("Loading dataset...")
df_full = pd.read_csv(DATA_PATH)
print(f"  Full dataset: {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")

df, _ = train_test_split(
    df_full, train_size=SAMPLE_SIZE,
    stratify=df_full[TARGET], random_state=RANDOM_STATE
)
df = df.reset_index(drop=True)
del df_full
print(f"  Working sample: {df.shape}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING
#     New features derived from raw positional data
# ═════════════════════════════════════════════════════════════════════════════
df['ball_speed']       = np.sqrt(df.ball_vel_x**2 + df.ball_vel_y**2 + df.ball_vel_z**2)
df['dist_to_goal_A']   = np.sqrt(df.ball_pos_x**2 + (df.ball_pos_y - GOAL_A_Y)**2 + df.ball_pos_z**2)
df['dist_to_goal_B']   = np.sqrt(df.ball_pos_x**2 + (df.ball_pos_y - GOAL_B_Y)**2 + df.ball_pos_z**2)
df['team_A_avg_boost'] = df[['p0_boost', 'p1_boost', 'p2_boost']].mean(axis=1)
df['team_B_avg_boost'] = df[['p3_boost', 'p4_boost', 'p5_boost']].mean(axis=1)
df['team_A_avg_y']     = df[['p0_pos_y', 'p1_pos_y', 'p2_pos_y']].mean(axis=1)
df['team_B_avg_y']     = df[['p3_pos_y', 'p4_pos_y', 'p5_pos_y']].mean(axis=1)

def min_dist_to_ball(df_, pids):
    cols = [np.sqrt((df_.ball_pos_x - df_[f'p{p}_pos_x'])**2 +
                    (df_.ball_pos_y - df_[f'p{p}_pos_y'])**2 +
                    (df_.ball_pos_z - df_[f'p{p}_pos_z'])**2) for p in pids]
    return pd.concat(cols, axis=1).min(axis=1)

df['team_A_min_dist'] = min_dist_to_ball(df, [0, 1, 2])
df['team_B_min_dist'] = min_dist_to_ball(df, [3, 4, 5])


# ═════════════════════════════════════════════════════════════════════════════
# 3.  DATA VISUALIZATIONS  (7 figures)
# ═════════════════════════════════════════════════════════════════════════════

# ── Fig 1: Class Distribution (both teams) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 1 – Scoring-Within-10 sec Class Distribution', fontsize=14, fontweight='bold')

for ax, col, team in zip(axes,
                          ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec'],
                          ['Team A', 'Team B']):
    counts = df[col].value_counts().sort_index()
    bars = ax.bar(['No Score (0)', 'Score (1)'], counts.values,
                  color=[BLUE, ORANGE], edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f'{val:,}\n({val / len(df) * 100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    ax.set_title(f'{team} – Scoring Within 10 Seconds', fontsize=12)
    ax.set_ylabel('Count')
    ax.set_ylim(0, counts.max() * 1.18)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis1_class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 2: Ball Position Heatmap (No Score vs Score) ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Fig 2 – Ball (X, Y) Position on Field: No Score vs Score (Team A)',
             fontsize=14, fontweight='bold')

no_score_sample = df[df[TARGET] == 0].sample(10_000, random_state=42)
score_sample    = df[df[TARGET] == 1].sample(min(5_000, (df[TARGET] == 1).sum()), random_state=42)

for ax, sub, label in zip(axes, [no_score_sample, score_sample], ['No Score', 'Score']):
    hb = ax.hexbin(sub.ball_pos_x, sub.ball_pos_y, gridsize=40, cmap='YlOrRd', mincnt=1)
    plt.colorbar(hb, ax=ax, label='Count')
    for y_line, lbl in [(GOAL_A_Y, 'Goal A'), (GOAL_B_Y, 'Goal B')]:
        ax.axhline(y_line, color='dodgerblue', lw=2.5, ls='--', alpha=0.8, label=lbl)
    ax.set_xlabel('Ball X'); ax.set_ylabel('Ball Y')
    ax.set_title(f'Ball Position – {label}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis2_ball_position_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 3: Ball Speed KDE – Score vs No Score ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Fig 3 – Ball Speed Distribution: Score vs No Score (Team A)',
             fontsize=14, fontweight='bold')

for flag, color, label in [(0, BLUE, 'No Score'), (1, ORANGE, 'Score (Team A)')]:
    df[df[TARGET] == flag]['ball_speed'].plot.kde(ax=ax, color=color, lw=2.5, label=label)

ax.set_xlabel('Ball Speed (game units / sec)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis3_ball_speed_kde.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 4: Ball Distance to Goal vs Scoring Probability ──────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Fig 4 – Ball Distance to Goal vs Scoring Probability',
             fontsize=14, fontweight='bold')

for ax, dist_col, tgt_col, team in zip(
        axes,
        ['dist_to_goal_A', 'dist_to_goal_B'],
        ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec'],
        ['Team A', 'Team B']):
    df['_bin'] = pd.cut(df[dist_col], bins=25)
    prob = (df.groupby('_bin', observed=True)[tgt_col]
              .mean().reset_index())
    prob['mid'] = prob['_bin'].apply(lambda x: x.mid)
    ax.plot(prob['mid'], prob[tgt_col] * 100, marker='o',
            color=ORANGE if team == 'Team A' else BLUE, lw=2)
    ax.fill_between(prob['mid'], prob[tgt_col] * 100, alpha=0.15,
                    color=ORANGE if team == 'Team A' else BLUE)
    ax.set_xlabel(f'Ball Distance to {team} Goal')
    ax.set_ylabel('Scoring Probability (%)')
    ax.set_title(f'{team} – Scoring Prob. vs Distance to Goal')

df.drop(columns='_bin', errors='ignore', inplace=True)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis4_distance_vs_probability.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 5: Event Time vs Scoring Probability ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Fig 5 – Time Remaining vs Scoring Probability (Both Teams)',
             fontsize=14, fontweight='bold')

df['_tbin'] = pd.cut(df['event_time'], bins=30)
for tgt_col, color, label in [
        ('team_A_scoring_within_10sec', BLUE,   'Team A'),
        ('team_B_scoring_within_10sec', ORANGE, 'Team B')]:
    prob = df.groupby('_tbin', observed=True)[tgt_col].mean().reset_index()
    prob['mid'] = prob['_tbin'].apply(lambda x: x.mid)
    ax.plot(prob['mid'], prob[tgt_col] * 100, marker='o',
            color=color, lw=2.5, label=label)

ax.set_xlabel('Time Before Event Ended (seconds) — decreasing = closer to goal')
ax.set_ylabel('Scoring Probability (%)')
ax.invert_xaxis()
ax.legend()
df.drop(columns='_tbin', errors='ignore', inplace=True)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis5_time_vs_probability.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 6: Team Boost Boxplot by Scoring Outcome ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Fig 6 – Team Average Boost by Scoring Outcome',
             fontsize=14, fontweight='bold')

for ax, boost_col, tgt_col, team in zip(
        axes,
        ['team_A_avg_boost', 'team_B_avg_boost'],
        ['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec'],
        ['Team A', 'Team B']):
    tmp = df[[boost_col, tgt_col]].copy()
    tmp[tgt_col] = tmp[tgt_col].map({0: 'No Score', 1: 'Score'})
    sns.boxplot(data=tmp, x=tgt_col, y=boost_col, ax=ax,
                palette={'No Score': BLUE, 'Score': ORANGE},
                order=['No Score', 'Score'])
    ax.set_title(f'{team} – Average Boost')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Average Boost (0–100)')

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis6_boost_by_outcome.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Fig 7: Ball Height (Z) Distribution by Scoring Outcome ──────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Fig 7 – Ball Height (Z) Distribution: Score vs No Score (Team A)',
             fontsize=14, fontweight='bold')

for flag, color, label in [(0, BLUE, 'No Score'), (1, ORANGE, 'Score (Team A)')]:
    df[df[TARGET] == flag]['ball_pos_z'].plot.kde(ax=ax, color=color, lw=2.5, label=label)

ax.axvline(df['ball_pos_z'].mean(), ls='--', color='gray', alpha=0.7, label='Overall Mean Z')
ax.set_xlabel('Ball Height (Z)')
ax.set_ylabel('Density')
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_vis7_ball_height.png', dpi=150, bbox_inches='tight')
plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PREPARE ML DATA
#     Techniques: median imputation for demolished-player NaNs, StandardScaler
# ═════════════════════════════════════════════════════════════════════════════
FEATURES = [
    'ball_pos_x', 'ball_pos_y', 'ball_pos_z',
    'ball_vel_x', 'ball_vel_y', 'ball_vel_z',
    'ball_speed', 'dist_to_goal_A', 'dist_to_goal_B',
    'team_A_avg_boost', 'team_B_avg_boost',
    'team_A_avg_y',     'team_B_avg_y',
    'team_A_min_dist',  'team_B_min_dist',
    'event_time',
    'p0_boost', 'p1_boost', 'p2_boost',
    'p3_boost', 'p4_boost', 'p5_boost',
    'boost0_timer', 'boost1_timer', 'boost2_timer',
    'boost3_timer', 'boost4_timer', 'boost5_timer',
]

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(df[FEATURES]), columns=FEATURES)
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"Class balance in train – 0: {(y_train==0).sum():,}  "
      f"1: {(y_train==1).sum():,}  "
      f"({(y_train==1).mean()*100:.1f}% positive)\n")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  DECISION TREE
# ═════════════════════════════════════════════════════════════════════════════
DT_PARAMS = dict(
    max_depth=8,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',   # handles class imbalance (~94/6 split)
    criterion='gini',
    random_state=RANDOM_STATE,
)
print("── Decision Tree ──")
print(f"  Hyperparameters: {DT_PARAMS}")

dt = DecisionTreeClassifier(**DT_PARAMS)
dt.fit(X_train, y_train)

y_pred_dt    = dt.predict(X_test)
y_prob_dt    = dt.predict_proba(X_test)[:, 1]
dt_acc       = accuracy_score(y_test, y_pred_dt)
dt_cm        = confusion_matrix(y_test, y_pred_dt)
dt_auc       = roc_auc_score(y_test, y_prob_dt)

print(f"  Test Accuracy : {dt_acc:.4f}")
print(f"  ROC-AUC       : {dt_auc:.4f}")
print(classification_report(y_test, y_pred_dt, target_names=['No Score', 'Score']))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
dt_cv_scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {dt_cv_scores.mean():.4f} ± {dt_cv_scores.std():.4f}\n")

# ── Decision Tree diagram (capped at depth 3 for readability) ────────────────
fig, ax = plt.subplots(figsize=(26, 10))
plot_tree(dt, max_depth=3, feature_names=FEATURES,
          class_names=['No Score', 'Score'],
          filled=True, rounded=True, ax=ax, fontsize=7,
          impurity=False, proportion=True)
ax.set_title('Decision Tree – Top 3 Levels (full tree depth = 8)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_dt_tree_diagram.png', dpi=120, bbox_inches='tight')
plt.show()

# ── Feature Importance ────────────────────────────────────────────────────────
importances = (pd.Series(dt.feature_importances_, index=FEATURES)
                 .sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(12, 7))
importances.head(15).sort_values().plot.barh(ax=ax, color=BLUE, edgecolor='white')
ax.set_title('Top 15 Feature Importances – Decision Tree (Gini)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Gini Importance')
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_dt_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 6.  LOGISTIC REGRESSION  (cross-validation comparison model)
# ═════════════════════════════════════════════════════════════════════════════
LR_PARAMS = dict(
    C=0.1,
    max_iter=1000,
    class_weight='balanced',
    solver='lbfgs',
    random_state=RANDOM_STATE,
)
print("── Logistic Regression ──")
print(f"  Hyperparameters: {LR_PARAMS}")

lr = LogisticRegression(**LR_PARAMS)
lr.fit(X_train_sc, y_train)

y_pred_lr    = lr.predict(X_test_sc)
y_prob_lr    = lr.predict_proba(X_test_sc)[:, 1]
lr_acc       = accuracy_score(y_test, y_pred_lr)
lr_cm        = confusion_matrix(y_test, y_pred_lr)
lr_auc       = roc_auc_score(y_test, y_prob_lr)

print(f"  Test Accuracy : {lr_acc:.4f}")
print(f"  ROC-AUC       : {lr_auc:.4f}")
print(classification_report(y_test, y_pred_lr, target_names=['No Score', 'Score']))

X_sc_full    = scaler.transform(X)
lr_cv_scores = cross_val_score(lr, X_sc_full, y, cv=cv, scoring='accuracy')
print(f"  5-Fold CV Accuracy: {lr_cv_scores.mean():.4f} ± {lr_cv_scores.std():.4f}\n")


# ═════════════════════════════════════════════════════════════════════════════
# 7.  COMPARISON PLOTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Test Accuracy + CV Scores ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Accuracy Comparison', fontsize=14, fontweight='bold')

# Left: test accuracy bar chart
ax = axes[0]
bars = ax.bar(['Decision Tree', 'Logistic\nRegression'],
              [dt_acc * 100, lr_acc * 100],
              color=[BLUE, ORANGE], width=0.45, edgecolor='white')
for bar, acc in zip(bars, [dt_acc, lr_acc]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{acc * 100:.2f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold')
ax.set_ylim(50, 100)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Test Set Accuracy')

# Right: CV scores per fold
ax = axes[1]
folds = [f'Fold {i+1}' for i in range(5)]
ax.plot(folds, dt_cv_scores * 100, marker='o', color=BLUE,  lw=2.5, label='Decision Tree')
ax.plot(folds, lr_cv_scores * 100, marker='s', color=ORANGE, lw=2.5, label='Logistic Regression')
ax.axhline(dt_cv_scores.mean() * 100, ls='--', color=BLUE,   alpha=0.5, lw=1.5)
ax.axhline(lr_cv_scores.mean() * 100, ls='--', color=ORANGE, alpha=0.5, lw=1.5)
ax.set_ylabel('CV Accuracy (%)')
ax.set_title('5-Fold Cross-Validation Accuracy')
ax.legend()

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_accuracy_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Confusion Matrices ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, cm, title in zip(axes,
                          [dt_cm, lr_cm],
                          ['Decision Tree', 'Logistic Regression']):
    ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=['No Score', 'Score']).plot(
        ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(title, fontsize=12)

plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ── ROC Curves ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.50)')

for y_prob, color, label in [(y_prob_dt, BLUE,   'Decision Tree'),
                              (y_prob_lr, ORANGE, 'Logistic Regression')]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{label}  (AUC = {auc_val:.3f})')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves – Decision Tree vs Logistic Regression',
             fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Cross-Validation Error Rates per Fold ────────────────────────────────────
dt_cv_err = 1 - dt_cv_scores
lr_cv_err = 1 - lr_cv_scores

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(5)
w = 0.35
ax.bar(x - w / 2, dt_cv_err * 100, w, label='Decision Tree',      color=BLUE,   edgecolor='white')
ax.bar(x + w / 2, lr_cv_err * 100, w, label='Logistic Regression', color=ORANGE, edgecolor='white')
ax.axhline(dt_cv_err.mean() * 100, ls='--', color=BLUE,   lw=1.5, alpha=0.7,
           label=f'DT Mean Error {dt_cv_err.mean()*100:.2f}%')
ax.axhline(lr_cv_err.mean() * 100, ls='--', color=ORANGE, lw=1.5, alpha=0.7,
           label=f'LR Mean Error {lr_cv_err.mean()*100:.2f}%')
ax.set_xticks(x)
ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_xlabel('Fold')
ax.set_ylabel('Error Rate (%)')
ax.set_title('Cross-Validation Error Rate per Fold',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{SAVE_DIR}rl_cv_error_rates.png', dpi=150, bbox_inches='tight')
plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# 8.  FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"\nResearch Question: Can we accurately predict that Team A")
print(f"will score within the next 10 seconds of game time?")
print()
print(f"Target variable  : {TARGET}")
print(f"Feature count    : {len(FEATURES)}")
print(f"Sample size      : {SAMPLE_SIZE:,}  (stratified from 2,149,381 rows)")
print(f"Train / Test     : {len(X_train):,} / {len(X_test):,}")
print()
print("─── Decision Tree ───────────────────────────────────────")
print(f"  Hyperparameters : {DT_PARAMS}")
print(f"  Test Accuracy   : {dt_acc*100:.2f}%")
print(f"  ROC-AUC         : {dt_auc:.4f}")
print(f"  CV Accuracy     : {dt_cv_scores.mean()*100:.2f}% ± {dt_cv_scores.std()*100:.2f}%")
print(f"  CV Error Rate   : {dt_cv_err.mean()*100:.2f}% ± {dt_cv_err.std()*100:.2f}%")
print()
print("─── Logistic Regression ─────────────────────────────────")
print(f"  Hyperparameters : {LR_PARAMS}")
print(f"  Test Accuracy   : {lr_acc*100:.2f}%")
print(f"  ROC-AUC         : {lr_auc:.4f}")
print(f"  CV Accuracy     : {lr_cv_scores.mean()*100:.2f}% ± {lr_cv_scores.std()*100:.2f}%")
print(f"  CV Error Rate   : {lr_cv_err.mean()*100:.2f}% ± {lr_cv_err.std()*100:.2f}%")
print()
print("─── Techniques / Data Handling ──────────────────────────")
print("  1. Stratified sampling (150k rows) to preserve class ratio")
print("  2. Feature engineering: ball_speed, dist_to_goal, team avg")
print("     boost/position, min player-to-ball distance per team")
print("  3. Median imputation for NaNs (demolished players)")
print("  4. StandardScaler applied before Logistic Regression only")
print("  5. class_weight='balanced' in both models (6% positive class)")
print("  6. StratifiedKFold(5) cross-validation")
print()
print(f"  All figures saved to {SAVE_DIR}")
