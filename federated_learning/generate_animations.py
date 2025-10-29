import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from federated_learning.fl_output import FLOutput

def main():
    print("=== Federated Learning Animation Generator ===")
    metrics_path = input("Enter the path to your metrics JSON file:\n").strip()
    if not os.path.isfile(metrics_path):
        print(f"Error: Metrics file not found at '{metrics_path}'")
        return

    output_dir = os.path.dirname(metrics_path)

    # Accuracy animation
    accuracy_anim_file = os.path.join(output_dir, "accuracy_progress.gif")
    print(f"Generating accuracy animation: {accuracy_anim_file}")
    FLOutput.animate_accuracy_progress(metrics_path, save_path=accuracy_anim_file)

    # Client participation animation
    participation_anim_file = os.path.join(output_dir, "client_participation.gif")
    print(f"Generating client participation animation: {participation_anim_file}")
    FLOutput.animate_client_participation(metrics_path, save_path=participation_anim_file)

    print("\nAnimations saved to:")
    print("Accuracy animation:", accuracy_anim_file)
    print("Client participation animation:", participation_anim_file)

if __name__ == "__main__":
    main()