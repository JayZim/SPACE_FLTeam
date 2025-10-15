"""
Filename: fl_output.py
Description: Handles evaluation, metrics computation, and logging for trained FL models
Author: Gagandeep Singh
Date: 2025-05-08
Version: 2.0
Python Version: 3.10.0

Changelog:
- 2024-08-07: Initial creation.
- 2025-05-08: Major update to support PyTorch model evaluation and metrics:
              - Added model evaluation functionality
              - Implemented accuracy and timing metrics
              - Added JSON and text logging capabilities
              - Included standalone test functionality
              - Enhanced error handling
- 2025-05-09: Added custom metrics and saved results to files

Usage:
1. Instantiate FLOutput with a test dataset
2. Pass a trained model for evaluation
3. Access metrics or log results as needed

Example:
    from fl_output import FLOutput
    output = FLOutput()
    output.evaluate_model(trained_model)
    accuracy = output.get_result()
    output.log_result("results.txt")
    output.write_to_file("results.json", format="json")
"""

import json
import time
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interfaces.output import Output


class FLOutput(Output):
    """
    Handles evaluation, metrics computation and result logging for Federated Learning models.
    Implements the Output interface to maintain compatibility with the modular system.
    """

    def __init__(self, test_dataset=None, batch_size=32):
        """
        Initialize the output module with an optional test dataset.

        Args:
            test_dataset: Optional PyTorch dataset to use for evaluation.
                          If None, the MNIST test dataset will be loaded by default.
            batch_size: Batch size to use for model evaluation
        """
        self.model = None
        self.metrics = {
            "accuracy": None,
            "loss": None,
            "processing_time": None,
            "evaluation_time": None,
            "timestamp": None,
            "additional_metrics": {}
        }
        self.batch_size = batch_size
        self.test_loader = None

        # Initialize test dataset if not provided
        if test_dataset is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # If you normalized in training, use the same stats here.
                # transforms.Normalize(mean=[0.3443, 0.3804, 0.4086], std=[0.1814, 0.1535, 0.1311])
            ])
            self.test_dataset = datasets.EuroSAT('~/.pytorch/EuroSAT/', download=True,
                                                transform=transform)
            self.test_loader = DataLoader(self.test_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False)
        else:
            self.test_dataset = test_dataset
            self.test_loader = DataLoader(self.test_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False)
        # Original Code for Safety Concern
        # if test_dataset is None:
        #     transform = transforms.Compose([transforms.ToTensor()])
        #     self.test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/',
        #                                       download=True,
        #                                       train=False,
        #                                       transform=transform)
        #     self.test_loader = DataLoader(self.test_dataset,
        #                                  batch_size=self.batch_size,
        #                                  shuffle=False)
        # else:
        #     self.test_dataset = test_dataset
        #     self.test_loader = DataLoader(self.test_dataset,
        #                                  batch_size=self.batch_size,
        #                                  shuffle=False)

        print(f"FLOutput initialized with {len(self.test_dataset)} test samples")

    def evaluate_model(self, model: nn.Module, processing_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a PyTorch model on the test dataset and compute metrics.

        Args:
            model: The PyTorch model to evaluate
            processing_time: Optional processing time (training time) to record

        Returns:
            Dictionary containing the computed metrics

        Raises:
            ValueError: If model is None or not a PyTorch model
        """
        if model is None:
            raise ValueError("Model cannot be None")

        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")

        self.model = model
        self.model.eval()  # Set model to evaluation mode

        # Record timestamp
        self.metrics["timestamp"] = datetime.now().isoformat()

        # Record processing time if provided
        if processing_time is not None:
            self.metrics["processing_time"] = processing_time

        # Start evaluation timing
        eval_start_time = time.time()

        # Compute metrics on test dataset
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():  # Disable gradient computation for evaluation
            for data, target in self.test_loader:
                output = self.model(data)
                loss = criterion(output, target)

                # Compute accuracy
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Accumulate loss
                running_loss += loss.item() * data.size(0)

        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = running_loss / total

        # Record metrics
        self.metrics["accuracy"] = accuracy
        self.metrics["loss"] = avg_loss
        self.metrics["evaluation_time"] = time.time() - eval_start_time

        print(f"Model evaluated: Accuracy = {accuracy:.2f}%, Loss = {avg_loss:.4f}")
        return self.metrics

    def set_result(self, metrics: Dict[str, Any]) -> None:
        """
        Set metrics results directly (useful when metrics are computed externally).

        Args:
            metrics: Dictionary containing metrics to set
        """
        if not isinstance(metrics, dict):
            raise TypeError("Metrics must be a dictionary")

        self.metrics.update(metrics)

    def get_result(self) -> Dict[str, Any]:
        """
        Get the current metrics results.

        Returns:
            Dictionary containing all computed metrics
        """
        return self.metrics

    def add_metric(self, name: str, value: Any) -> None:
        """
        Add a custom metric to the additional_metrics dictionary.

        Args:
            name: Name of the metric
            value: Value of the metric
        """
        self.metrics["additional_metrics"][name] = value

    def log_result(self, path: Optional[str] = None) -> None:
        """
        Log current metrics to console and optionally to a file.

        Args:
            path: Optional path to a log file. If provided, metrics are appended to the file.
        """
        if self.metrics["accuracy"] is None:
            print("Warning: No metrics available to log. Evaluate a model first.")
            return

        # Format log message
        log_message = f"--- Federated Learning Results ({self.metrics['timestamp']}) ---\n"
        log_message += f"Accuracy: {self.metrics['accuracy']:.2f}%\n"
        log_message += f"Loss: {self.metrics['loss']:.4f}\n"

        if self.metrics["processing_time"] is not None:
            log_message += f"Processing Time: {self.metrics['processing_time']:.2f} seconds\n"

        log_message += f"Evaluation Time: {self.metrics['evaluation_time']:.2f} seconds\n"

        # Include additional metrics
        if self.metrics["additional_metrics"]:
            log_message += "Additional Metrics:\n"
            for name, value in self.metrics["additional_metrics"].items():
                log_message += f"  {name}: {value}\n"

        log_message += "------------------------------------------------\n"

        # Print to console
        print(log_message)

        # Write to file if path is provided
        if path is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

                # Append to file
                with open(path, 'a') as f:
                    f.write(log_message)
                print(f"Results logged to {path}")
            except Exception as e:
                print(f"Error writing log to file: {e}")

    def write_to_file(self, path: str, format: str = "json") -> None:
        """
        Write metrics to a file in the specified format.

        Args:
            path: Path to the output file
            format: Format of the output file. Options: "json", "txt"

        Raises:
            ValueError: If format is not supported
        """
        if self.metrics["accuracy"] is None:
            print("Warning: No metrics available to write. Evaluate a model first.")
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            if format.lower() == "json":
                with open(path, 'w') as f:
                    json.dump(self.metrics, f, indent=4, default=str)
                print(f"Results written to {path} in JSON format")

            elif format.lower() == "txt":
                with open(path, 'w') as f:
                    for key, value in self.metrics.items():
                        if key == "additional_metrics":
                            f.write(f"{key}:\n")
                            for sub_key, sub_value in value.items():
                                f.write(f"  {sub_key}: {sub_value}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                print(f"Results written to {path} in text format")

            else:
                raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'.")

        except Exception as e:
            print(f"Error writing to file: {e}")

    def compute_confusion_matrix(self, num_classes: int = 10) -> np.ndarray:
        """
        Compute confusion matrix for the evaluated model.

        Args:
            num_classes: Number of classes in the classification task

        Returns:
            Confusion matrix as a numpy array

        Raises:
            ValueError: If model has not been evaluated yet
        """
        if self.model is None:
            raise ValueError("No model has been evaluated yet")

        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        self.model.eval()
        with torch.no_grad():
            for data, target in self.test_loader:
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)

                # Update confusion matrix
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    confusion_matrix[t.item(), p.item()] += 1

        # Add to additional metrics
        self.metrics["additional_metrics"]["confusion_matrix"] = confusion_matrix.tolist()
        return confusion_matrix

    def compute_per_class_metrics(self, num_classes: int = 10) -> Dict[str, List[float]]:
        """
        Compute precision, recall, and F1-score for each class.

        Args:
            num_classes: Number of classes in the classification task

        Returns:
            Dictionary containing per-class metrics

        Raises:
            ValueError: If model has not been evaluated yet
        """
        if self.model is None:
            raise ValueError("No model has been evaluated yet")

        # Compute confusion matrix if not already computed
        if "confusion_matrix" not in self.metrics["additional_metrics"]:
            confusion_matrix = self.compute_confusion_matrix(num_classes)
        else:
            confusion_matrix = np.array(self.metrics["additional_metrics"]["confusion_matrix"])

        # Initialize per-class metrics
        precision = np.zeros(num_classes)
        recall = np.zeros(num_classes)
        f1_score = np.zeros(num_classes)

        # Compute metrics for each class
        for i in range(num_classes):
            # Precision: TP / (TP + FP)
            precision[i] = confusion_matrix[i, i] / max(np.sum(confusion_matrix[:, i]), 1)

            # Recall: TP / (TP + FN)
            recall[i] = confusion_matrix[i, i] / max(np.sum(confusion_matrix[i, :]), 1)

            # F1-score: 2 * (precision * recall) / (precision + recall)
            if precision[i] + recall[i] > 0:
                f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
            else:
                f1_score[i] = 0

        # Add to additional metrics
        per_class_metrics = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1_score": f1_score.tolist()
        }

        self.metrics["additional_metrics"]["per_class_metrics"] = per_class_metrics
        return per_class_metrics

    def save_model(self, path: str) -> None:
        """
        Save the evaluated model to a file.

        Args:
            path: Path to save the model

        Raises:
            ValueError: If no model has been evaluated yet
        """
        if self.model is None:
            raise ValueError("No model has been evaluated yet")

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)

            # Save model
            torch.save(self.model.state_dict(), path)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def animate_client_participation(metrics_json_path, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import numpy as np
        import json

        with open(metrics_json_path, "r") as f:
            metrics = json.load(f)

        participation_log = metrics["additional_metrics"]["participation_log"]
        num_timesteps = len(participation_log)
        num_clients = max(
            max([entry["aggregation_server"] if entry["aggregation_server"] is not None else 0] +
                [entry["redistribution_server"] if entry["redistribution_server"] is not None else 0] +
                entry["in_range_clients"] + entry["out_of_range_clients"])
            for entry in participation_log
        ) + 1

        # Preprocess: Replace zeros with previous non-zero accuracy for skipped rounds
        processed_accuracies = []
        last_acc = None
        for entry in participation_log:
            acc = entry.get("accuracy", None)
            if acc == 0 or acc is None:
                processed_accuracies.append(last_acc if last_acc is not None else 0)
            else:
                processed_accuracies.append(acc)
                last_acc = acc

        angles = np.linspace(0, 2 * np.pi, num_clients, endpoint=False)
        radius = 5
        client_positions = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]

        # Sliding window parameters
        window_size = 50
        col_size = window_size // 2

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 3])
        ax_timeline1 = fig.add_subplot(gs[0, 0])
        ax_timeline2 = fig.add_subplot(gs[0, 1])
        ax = fig.add_subplot(gs[0, 2])
        ax.set_xlim(-radius-2, radius+2)
        ax.set_ylim(-radius-2.5, radius+2)
        ax.set_aspect('equal')
        ax.axis('off')

        circles = []
        arrows = []
        phase_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, fontsize=16, ha="center")
        accuracy_text = ax.text(0.5, -0.18, '', transform=ax.transAxes, fontsize=14, ha="center")
        round_acc_text = ax.text(0.5, 0.95, '', transform=ax.transAxes, fontsize=14, ha="center", color="purple")

        def init():
            for i, (x, y) in enumerate(client_positions):
                circle = plt.Circle((x, y), 0.6, color='gray', alpha=0.5)
                ax.add_patch(circle)
                ax.text(x, y, f"{i+1}", fontsize=12, ha='center', va='center')
                circles.append(circle)
            return circles + [phase_text, accuracy_text, round_acc_text]

        def update(frame):
            for arrow in arrows:
                arrow.remove()
            arrows.clear()

            data = participation_log[frame]
            phase = data['phase']
            agg_server = data['aggregation_server']
            redist_server = data['redistribution_server']
            in_range = data['in_range_clients']
            out_range = data['out_of_range_clients']
            acc = data['accuracy']
            round_num = data.get('round', None)
            processed_acc = processed_accuracies[frame]

            # --- Sliding window logic ---
            # Determine which window we're in
            window_start = (frame // window_size) * window_size
            window_end = min(window_start + window_size, num_timesteps)
            col1_start = window_start
            col1_end = min(window_start + col_size, window_end)
            col2_start = col1_end
            col2_end = window_end

            # --- Draw timeline column 1 ---
            ax_timeline1.clear()
            ax_timeline1.set_ylim(col1_end - col1_start + 1, 0)
            ax_timeline1.set_xlim(0, 1)
            ax_timeline1.axis('off')
            for idx, t in enumerate(range(col1_start, col1_end)):
                entry = participation_log[t]
                phase_label = entry['phase']
                acc_val = processed_accuracies[t]
                if t == frame:
                    ax_timeline1.text(0.05, idx+1, f"Timestep {entry['timestep']}: {phase_label}", fontsize=12, color='blue', weight='bold', va='center')
                    acc_str = f"{acc_val:.2%}" if acc_val != 0 else "Training Skipped"
                    ax_timeline1.text(0.05, idx+1.5, f"Accuracy: {acc_str}", fontsize=12, color='red', weight='bold', va='center')
                else:
                    ax_timeline1.text(0.05, idx+1, f"Timestep {entry['timestep']}: {phase_label}", fontsize=10, color='gray', va='center')
                    acc_str = f"{acc_val:.2%}" if acc_val != 0 else "Training Skipped"
                    ax_timeline1.text(0.05, idx+1.5, f"Accuracy: {acc_str}", fontsize=9, color='gray', va='center')

            # --- Draw timeline column 2 ---
            ax_timeline2.clear()
            ax_timeline2.set_ylim(col2_end - col2_start + 1, 0)
            ax_timeline2.set_xlim(0, 1)
            ax_timeline2.axis('off')
            for idx, t in enumerate(range(col2_start, col2_end)):
                entry = participation_log[t]
                phase_label = entry['phase']
                acc_val = processed_accuracies[t]
                if t == frame:
                    ax_timeline2.text(0.05, idx+1, f"Timestep {entry['timestep']}: {phase_label}", fontsize=12, color='blue', weight='bold', va='center')
                    acc_str = f"{acc_val:.2%}" if acc_val != 0 else "Training Skipped"
                    ax_timeline2.text(0.05, idx+1.5, f"Accuracy: {acc_str}", fontsize=12, color='red', weight='bold', va='center')
                else:
                    ax_timeline2.text(0.05, idx+1, f"Timestep {entry['timestep']}: {phase_label}", fontsize=10, color='gray', va='center')
                    acc_str = f"{acc_val:.2%}" if acc_val != 0 else "Training Skipped"
                    ax_timeline2.text(0.05, idx+1.5, f"Accuracy: {acc_str}", fontsize=9, color='gray', va='center')

            # --- Draw participation diagram (same as before) ---
            for i, circle in enumerate(circles):
                if i == agg_server and phase == "TRANSMITTING":
                    circle.set_color('gold')
                    circle.set_alpha(1.0)
                    circle.set_edgecolor('red')
                    circle.set_linewidth(3)
                elif i == redist_server and phase == "REDISTRIBUTION":
                    circle.set_color('deepskyblue')
                    circle.set_alpha(1.0)
                    circle.set_edgecolor('blue')
                    circle.set_linewidth(3)
                elif i in in_range:
                    if phase == "REDISTRIBUTION":
                        circle.set_color('deepskyblue')
                        circle.set_alpha(0.8)
                    else:
                        circle.set_color('green')
                        circle.set_alpha(0.8)
                    circle.set_edgecolor('black')
                    circle.set_linewidth(1)
                else:
                    circle.set_color('gray')
                    circle.set_alpha(0.3)
                    circle.set_edgecolor('black')
                    circle.set_linewidth(1)

            if phase == "TRANSMITTING":
                for i in in_range:
                    if agg_server is not None and i != agg_server:
                        x0, y0 = client_positions[i]
                        x1, y1 = client_positions[agg_server]
                        arrow = ax.annotate("",
                                            xy=(x1, y1), xytext=(x0, y0),
                                            arrowprops=dict(arrowstyle="->", color='green', lw=2))
                        arrows.append(arrow)
            elif phase == "CHECK":
                if agg_server is not None and redist_server is not None and agg_server != redist_server:
                    x0, y0 = client_positions[agg_server]
                    x1, y1 = client_positions[redist_server]
                    arrow = ax.annotate("",
                                        xy=(x1, y1), xytext=(x0, y0),
                                        arrowprops=dict(arrowstyle="->", color='orange', lw=2, linestyle='dashed'))
                    arrows.append(arrow)
            elif phase == "REDISTRIBUTION":
                for i in in_range:
                    if redist_server is not None and i != redist_server:
                        x0, y0 = client_positions[redist_server]
                        x1, y1 = client_positions[i]
                        arrow = ax.annotate("",
                                            xy=(x1, y1), xytext=(x0, y0),
                                            arrowprops=dict(arrowstyle="->", color='deepskyblue', lw=2))
                        arrows.append(arrow)

            phase_label = f"Phase: {phase}"
            if phase == "CHECK":
                phase_label += " (Model transfer)"
            elif phase == "REDISTRIBUTION":
                phase_label += " (Global model distribution)"
            phase_text.set_text(phase_label)

            if acc is not None:
                accuracy_text.set_text(f"Timestep: {data['timestep']} | Accuracy: {processed_acc:.2%}")
            else:
                accuracy_text.set_text(f"Timestep: {data['timestep']} | Training Skipped")

            if round_num is not None:
                if acc is not None:
                    round_acc_text.set_text(f"Round: {round_num} | Accuracy: {processed_acc:.2%}")
                else:
                    round_acc_text.set_text(f"Round: {round_num} | Training Skipped")
            else:
                if acc is not None:
                    round_acc_text.set_text(f"Accuracy: {processed_acc:.2%}")
                else:
                    round_acc_text.set_text("Training Skipped")

            return circles + arrows + [phase_text, accuracy_text, round_acc_text]

        ani = animation.FuncAnimation(fig, update, frames=num_timesteps,
                                      init_func=init, blit=False, repeat=False)

        if save_path:
            ani.save(save_path, writer='pillow', fps=1)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()

    def animate_accuracy_progress(metrics_json_path, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        import json

        with open(metrics_json_path, "r") as f:
            metrics = json.load(f)

        additional_metrics = metrics["additional_metrics"]
        round_accuracies = additional_metrics.get("round_accuracies", [])
        timesteps = list(range(1, len(round_accuracies) + 1))

        # Preprocess: Replace zeros with previous non-zero accuracy for skipped rounds
        processed_accuracies = []
        last_acc = None
        for acc in round_accuracies:
            if acc == 0 or acc is None:
                # Use last non-zero accuracy if available
                processed_accuracies.append(last_acc if last_acc is not None else 0)
            else:
                processed_accuracies.append(acc)
                last_acc = acc

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlim(0, len(processed_accuracies) + 1)
        ax.set_ylim(0.00, 1.05)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Accuracy")
        ax.set_title("Federated Learning Accuracy Progression")

        def update(frame):
            ax.clear()
            ax.set_xlim(0, len(processed_accuracies) + 1)
            ax.set_ylim(0.00, 1.05)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Accuracy")
            ax.set_title("Federated Learning Accuracy Progression")
            ax.bar(timesteps[:frame+1], processed_accuracies[:frame+1], color="green", width=0.8)
            acc = round_accuracies[frame]
            if acc == 0 or acc is None:
                accuracy_text = ax.text(0.5, 0.95, f"Timestep: {frame+1} | Training Skipped", transform=ax.transAxes, fontsize=14, ha="center")
            else:
                accuracy_text = ax.text(0.5, 0.95, f"Timestep: {frame+1} | Accuracy: {acc:.2%}", transform=ax.transAxes, fontsize=14, ha="center")
            return [accuracy_text]

        ani = animation.FuncAnimation(fig, update, frames=len(processed_accuracies), blit=False, repeat=False)

        if save_path:
            ani.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()


def run_test():
    """
    Standalone test function to verify the FLOutput functionality.
    """
    print("Running standalone test for FLOutput...")

    # Create timestamped run folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join(os.path.dirname(__file__), "results_from_output")
    run_dir = os.path.join(results_root, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(28 * 28, 10)

        def forward(self, x):
            x = x.view(-1, 28 * 28)
            return self.fc(x)

    # Initialize model and output module
    model = SimpleModel()
    output = FLOutput()

    # Simulate processing time and evaluate model
    processing_time = 10.5  # seconds
    print("Evaluating model...")
    output.evaluate_model(model, processing_time)

    # Create output file paths in this run folder
    log_file = os.path.join(run_dir, f"output_test_{timestamp}.log")
    metrics_file = os.path.join(run_dir, f"output_test_{timestamp}.json")
    model_file = os.path.join(run_dir, f"output_test_{timestamp}.pt")

    # Log results
    print("\nLogging results...")
    output.log_result(log_file)

    # Add custom metrics
    output.add_metric("custom_metric_1", 0.95)
    output.add_metric("custom_metric_2", [1, 2, 3, 4])

    # Compute additional metrics
    print("\nComputing confusion matrix...")
    output.compute_confusion_matrix()

    print("\nComputing per-class metrics...")
    output.compute_per_class_metrics()

    # Write results to files
    print("\nWriting results to files...")
    output.write_to_file(metrics_file, "json")
    output.save_model(model_file)

    print("\nResults have been saved to:")
    print("Log file:", log_file)
    print("Metrics file:", metrics_file)
    print("Model file:", model_file)

    # Generate visualizations for this run
    from federated_learning.fl_visualization import FLVisualization
    print("\nGenerating visualizations...")
    viz = FLVisualization(results_dir=run_dir)
    viz.visualize_from_json(metrics_file)
    print(f"Visualizations saved under {run_dir}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    """Standalone entry point for testing FLOutput."""
    run_test()
