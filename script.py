import subprocess
import sys

def main():
    # Define the valid options you want to allow
    valid_options = [1, 2 , 3]  # Adding more valid options

    if len(sys.argv) != 2:
        print("Usage: python3 execute_cremi.py <number>")
        sys.exit(1)

    try:
        # Parse the argument and check if it's a valid option
        number = int(sys.argv[1])
        if number not in valid_options:
            print(f"Invalid choice. Please select one of {valid_options}.")
            sys.exit(1)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        sys.exit(1)

    # Define the execution orders based on the number input
    execution_order = {
        1: [  
            f"python train.py --config configs/config_CombinedLoss.yaml",
            f"python train.py --config configs/config_dice_loss.yaml ",
        ],
        2: [
            f"python train.py --config configs/config_focal_loss.yaml",
            f"python train.py --config configs/config_TverskyLoss.yaml ",
        ],
        3: [
            f"python train.py --config configs/config.yaml ",
        ]
    }

    # Get the list of commands for the selected option
    order = execution_order[number]

    # Execute each command in the selected set
    for command in order:
        print(f"Executing: {command}")
        process = subprocess.run(command, shell=True)
        if process.returncode != 0:
            # print(f"Command failed: {command}")
            print(f"Command failed (exit code {process.returncode}): {command}")
            print("Continuing to next command...")
            # sys.exit(1)

    print("All commands executed successfully.")

if __name__ == "__main__":
    main()