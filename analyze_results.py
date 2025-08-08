#!/usr/bin/env python
"""Analyze training results and display summary."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_metrics(metrics_path="checkpoints/training_metrics.json"):
    """Load training metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def print_summary(metrics):
    """Print training summary."""
    print("="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    # Training info
    train_metrics = metrics.get('train', [])
    val_metrics = metrics.get('val', [])
    test_metrics = metrics.get('test', [])
    
    print(f"\n📊 Training Details:")
    print(f"  • Epochs trained: {len(train_metrics)}")
    
    if train_metrics:
        final_train = train_metrics[-1]
        print(f"  • Final training loss: {final_train.get('loss', 'N/A'):.4f}")
        print(f"  • Final training accuracy: {final_train.get('accuracy', 0):.4f}")
        print(f"  • Final training AUC: {final_train.get('auc_macro', 0):.4f}")
    
    if val_metrics:
        # Find best validation metrics
        best_val_auc = max(val_metrics, key=lambda x: x.get('auc_macro', 0))
        best_val_acc = max(val_metrics, key=lambda x: x.get('accuracy', 0))
        
        print(f"\n🎯 Best Validation Results:")
        print(f"  • Best AUC: {best_val_auc.get('auc_macro', 0):.4f}")
        print(f"  • Best Accuracy: {best_val_acc.get('accuracy', 0):.4f}")
        print(f"  • Best Balanced Accuracy: {best_val_auc.get('balanced_accuracy', 0):.4f}")
    
    if test_metrics:
        test = test_metrics[-1]
        print(f"\n✅ Test Results:")
        print(f"  • Test AUC: {test.get('auc_macro', 0):.4f}")
        print(f"  • Test Accuracy: {test.get('accuracy', 0):.4f}")
        print(f"  • Test Balanced Accuracy: {test.get('balanced_accuracy', 0):.4f}")
        print(f"  • Test Sensitivity: {test.get('avg_sensitivity', 0):.4f}")
        print(f"  • Test Specificity: {test.get('avg_specificity', 0):.4f}")
        
        # Per-class results
        print(f"\n📋 Per-Class Performance:")
        classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        for cls in classes:
            f1 = test.get(f'{cls}_f1', 0)
            sensitivity = test.get(f'{cls}_sensitivity', 0)
            specificity = test.get(f'{cls}_specificity', 0)
            support = test.get(f'{cls}_support', 0)
            print(f"  {cls:8s}: F1={f1:.3f}, Sens={sensitivity:.3f}, Spec={specificity:.3f}, N={support}")

def plot_training_curves(metrics, save_path="training_curves.png"):
    """Plot training curves."""
    train_metrics = metrics.get('train', [])
    val_metrics = metrics.get('val', [])
    
    if not train_metrics or not val_metrics:
        print("No metrics to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    epochs = range(1, len(train_metrics) + 1)
    train_loss = [m.get('loss', 0) for m in train_metrics]
    val_loss = [m.get('loss', 0) for m in val_metrics]
    
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', marker='o', markersize=3)
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', marker='s', markersize=3)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy curve
    train_acc = [m.get('accuracy', 0) for m in train_metrics]
    val_acc = [m.get('accuracy', 0) for m in val_metrics]
    
    axes[0, 1].plot(epochs, train_acc, label='Train Accuracy', marker='o', markersize=3)
    axes[0, 1].plot(epochs, val_acc, label='Val Accuracy', marker='s', markersize=3)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # AUC curve
    train_auc = [m.get('auc_macro', 0) for m in train_metrics]
    val_auc = [m.get('auc_macro', 0) for m in val_metrics]
    
    axes[1, 0].plot(epochs, train_auc, label='Train AUC', marker='o', markersize=3)
    axes[1, 0].plot(epochs, val_auc, label='Val AUC', marker='s', markersize=3)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Training and Validation AUC-ROC')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Per-class F1 scores (validation)
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    final_val = val_metrics[-1]
    f1_scores = [final_val.get(f'{cls}_f1', 0) for cls in classes]
    
    axes[1, 1].bar(classes, f1_scores)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Final Validation F1 Scores by Class')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for i, (cls, f1) in enumerate(zip(classes, f1_scores)):
        axes[1, 1].text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 Training curves saved to: {save_path}")
    plt.show()

def check_saved_files():
    """Check what files were saved during training."""
    print("\n📁 Saved Files:")
    
    # Check checkpoints directory
    checkpoint_dir = Path("checkpoints")
    if checkpoint_dir.exists():
        files = list(checkpoint_dir.glob("*"))
        if files:
            print(f"\n  Checkpoints directory ({checkpoint_dir}):")
            for f in files:
                size = f.stat().st_size / 1024 / 1024  # MB
                print(f"    • {f.name}: {size:.2f} MB")
        else:
            print(f"  ⚠️  No files in checkpoints directory")
    
    # Check outputs directory
    outputs_dir = Path("outputs")
    if outputs_dir.exists():
        latest_dirs = sorted(outputs_dir.glob("*/*/*"), reverse=True)[:3]
        if latest_dirs:
            print(f"\n  Latest training outputs:")
            for d in latest_dirs:
                if d.is_dir():
                    log_file = d / "train.log"
                    if log_file.exists():
                        size = log_file.stat().st_size / 1024  # KB
                        print(f"    • {d.relative_to(outputs_dir)}: train.log ({size:.2f} KB)")
    
    # Check for model files
    model_files = list(Path(".").glob("**/*.pth"))
    if model_files:
        print(f"\n  Model files found:")
        for f in model_files[:5]:  # Show max 5
            size = f.stat().st_size / 1024 / 1024  # MB
            print(f"    • {f}: {size:.2f} MB")
    else:
        print(f"\n  ⚠️  No .pth model files found")
        print(f"  ℹ️  Model checkpoints should be saved during training")
        print(f"     You may need to retrain with checkpoint saving enabled")

def main():
    """Main function."""
    print("Skin Lesion Classification - Results Analysis")
    print("="*60)
    
    # Load metrics
    metrics_path = "checkpoints/training_metrics.json"
    if not Path(metrics_path).exists():
        print(f"❌ Metrics file not found: {metrics_path}")
        print("   Please run training first.")
        return
    
    metrics = load_metrics(metrics_path)
    
    # Print summary
    print_summary(metrics)
    
    # Check saved files
    check_saved_files()
    
    # Plot training curves
    try:
        plot_training_curves(metrics)
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
    
    print("\n" + "="*60)
    print("✅ Analysis complete!")
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    
    val_metrics = metrics.get('val', [])
    if val_metrics:
        best_auc = max(v.get('auc_macro', 0) for v in val_metrics)
        if best_auc > 0.97:
            print("  • Excellent performance achieved! (AUC > 0.97)")
            print("  • Model is ready for deployment")
        elif best_auc > 0.90:
            print("  • Good performance achieved (AUC > 0.90)")
            print("  • Consider training for more epochs or trying a larger model")
        else:
            print("  • Performance needs improvement")
            print("  • Try: more epochs, different architecture, or data augmentation")
    
    # Check for overfitting
    if train_metrics and val_metrics:
        train_acc = train_metrics[-1].get('accuracy', 0)
        val_acc = val_metrics[-1].get('accuracy', 0)
        if train_acc - val_acc > 0.1:
            print("  • ⚠️  Possible overfitting detected (train-val gap > 0.1)")
            print("  • Consider: more dropout, data augmentation, or regularization")

if __name__ == "__main__":
    main()