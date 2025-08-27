"""Metrics computation for binary classification evaluation."""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix,
    classification_report
)
from loguru import logger
import pandas as pd


class BinaryClassificationMetrics:
    """Compute and manage binary classification metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.detailed_results = []
    
    def compute_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[Optional[int]], 
        y_prob: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive binary classification metrics."""
        logger.info("Computing binary classification metrics")
        
        # Handle None predictions by treating them as wrong predictions
        y_pred_clean = []
        y_true_clean = []
        y_prob_clean = [] if y_prob else None
        
        for i, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
            if pred_label is not None:
                y_pred_clean.append(pred_label)
                y_true_clean.append(true_label)
                if y_prob:
                    y_prob_clean.append(y_prob[i])
        
        if len(y_pred_clean) == 0:
            logger.error("No valid predictions found")
            return {"error": "No valid predictions"}
        
        # Basic metrics
        accuracy = accuracy_score(y_true_clean, y_pred_clean)
        precision = precision_score(y_true_clean, y_pred_clean, average='binary', zero_division=0)
        recall = recall_score(y_true_clean, y_pred_clean, average='binary', zero_division=0)
        f1 = f1_score(y_true_clean, y_pred_clean, average='binary', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true_clean, y_pred_clean)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, len(y_true_clean))
        
        # Derived metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "specificity": float(specificity),
            "negative_predictive_value": float(npv),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "total_samples": len(y_true),
            "valid_predictions": len(y_pred_clean),
            "invalid_predictions": len(y_true) - len(y_pred_clean),
            "confusion_matrix": cm.tolist()
        }
        
        # ROC-AUC if probabilities are available
        if y_prob_clean and len(set(y_true_clean)) > 1:
            try:
                auc = roc_auc_score(y_true_clean, y_prob_clean)
                metrics["roc_auc"] = float(auc)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None
        
        # Classification report
        try:
            class_report = classification_report(
                y_true_clean, y_pred_clean, 
                target_names=['Negative', 'Positive'],
                output_dict=True,
                zero_division=0
            )
            metrics["classification_report"] = class_report
        except Exception as e:
            logger.warning(f"Could not generate classification report: {e}")
            metrics["classification_report"] = None
        
        self.metrics = metrics
        logger.info(f"Metrics computed: Accuracy={accuracy:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        return metrics
    
    def compute_detailed_results(
        self, 
        predictions: List[Dict[str, Any]], 
        true_labels: List[int]
    ) -> List[Dict[str, Any]]:
        """Compute detailed per-sample results."""
        logger.info("Computing detailed per-sample results")
        
        detailed_results = []
        
        for i, (pred_data, true_label) in enumerate(zip(predictions, true_labels)):
            result = {
                "sample_id": i,
                "input_text": pred_data.get("input_text", ""),
                "true_label": true_label,
                "predicted_label": pred_data.get("predicted_label"),
                "raw_response": pred_data.get("raw_response", ""),
                "is_correct": pred_data.get("predicted_label") == true_label if pred_data.get("predicted_label") is not None else False,
                "is_valid_prediction": pred_data.get("predicted_label") is not None,
                "prompt_tokens": pred_data.get("prompt_tokens", 0),
                "completion_tokens": pred_data.get("completion_tokens", 0),
                "total_tokens": pred_data.get("total_tokens", 0),
                "finish_reason": pred_data.get("finish_reason", "")
            }
            detailed_results.append(result)
        
        self.detailed_results = detailed_results
        return detailed_results
    
    def save_results(self, output_dir: Path, save_detailed: bool = True) -> None:
        """Save metrics and detailed results to files."""
        logger.info(f"Saving results to {output_dir}")
        
        # Save main metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        
        # Save detailed results if requested
        if save_detailed and self.detailed_results:
            # JSON format
            detailed_file = output_dir / "detailed_results.json"
            with open(detailed_file, 'w') as f:
                json.dump(self.detailed_results, f, indent=2, default=str)
            
            # CSV format for easier analysis
            detailed_csv = output_dir / "detailed_results.csv"
            df = pd.DataFrame(self.detailed_results)
            df.to_csv(detailed_csv, index=False)
        
        logger.info(f"Results saved to {output_dir}")
    
    def print_summary(self) -> None:
        """Print a summary of the metrics using Rich."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        if not self.metrics:
            logger.warning("No metrics to display")
            return
        
        # Create main metrics table
        table = Table(title="Binary Classification Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="bold cyan", width=20)
        table.add_column("Value", style="green", justify="right")
        table.add_column("Description", style="dim")
        
        # Sample statistics
        table.add_row("Total Samples", str(self.metrics['total_samples']), "Number of samples in dataset")
        table.add_row("Valid Predictions", str(self.metrics['valid_predictions']), "Successfully parsed predictions")
        table.add_row("Invalid Predictions", str(self.metrics['invalid_predictions']), "Failed to parse model response")
        table.add_row("", "", "")  # Separator
        
        if self.metrics['valid_predictions'] > 0:
            # Core metrics
            table.add_row("Accuracy", f"{self.metrics['accuracy']:.4f}", "Overall correct predictions")
            table.add_row("Precision", f"{self.metrics['precision']:.4f}", "True pos / (True pos + False pos)")
            table.add_row("Recall", f"{self.metrics['recall']:.4f}", "True pos / (True pos + False neg)")
            table.add_row("F1-Score", f"{self.metrics['f1_score']:.4f}", "Harmonic mean of precision & recall")
            table.add_row("Specificity", f"{self.metrics['specificity']:.4f}", "True neg / (True neg + False pos)")
            
            if self.metrics.get('roc_auc') is not None:
                table.add_row("ROC-AUC", f"{self.metrics['roc_auc']:.4f}", "Area under ROC curve")
        
        console.print(table)
        
        # Confusion matrix
        if self.metrics['valid_predictions'] > 0:
            cm_table = Table(title="Confusion Matrix", show_header=True, header_style="bold blue")
            cm_table.add_column("", style="bold")
            cm_table.add_column("Predicted: Negative", style="cyan", justify="center")
            cm_table.add_column("Predicted: Positive", style="yellow", justify="center")
            
            cm_table.add_row(
                "Actual: Negative", 
                f"[green]{self.metrics['true_negatives']}[/green] (TN)",
                f"[red]{self.metrics['false_positives']}[/red] (FP)"
            )
            cm_table.add_row(
                "Actual: Positive",
                f"[red]{self.metrics['false_negatives']}[/red] (FN)", 
                f"[green]{self.metrics['true_positives']}[/green] (TP)"
            )
            
            console.print(cm_table)
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """Analyze common error patterns."""
        if not self.detailed_results:
            return {}
        
        logger.info("Performing error analysis")
        
        # Get incorrect predictions
        errors = [r for r in self.detailed_results if not r['is_correct']]
        invalid_preds = [r for r in self.detailed_results if not r['is_valid_prediction']]
        
        analysis = {
            "total_errors": len(errors),
            "invalid_predictions": len(invalid_preds),
            "false_positives": len([e for e in errors if e['true_label'] == 0 and e['predicted_label'] == 1]),
            "false_negatives": len([e for e in errors if e['true_label'] == 1 and e['predicted_label'] == 0]),
        }
        
        # Sample error cases
        if errors:
            analysis["sample_false_positives"] = [
                {"text": e["input_text"][:200], "response": e["raw_response"]} 
                for e in errors if e['true_label'] == 0 and e['predicted_label'] == 1
            ][:5]
            
            analysis["sample_false_negatives"] = [
                {"text": e["input_text"][:200], "response": e["raw_response"]} 
                for e in errors if e['true_label'] == 1 and e['predicted_label'] == 0
            ][:5]
        
        if invalid_preds:
            analysis["sample_invalid_predictions"] = [
                {"text": e["input_text"][:200], "response": e["raw_response"]} 
                for e in invalid_preds
            ][:5]
        
        return analysis