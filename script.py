import subprocess
import sys

def main():
    # Define the valid options you want to allow
    valid_options = [1, 2 , 3,4,5,6,7,8,9,10]  # Adding more valid options

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
            f"python train.py --config configs/config_dice_loss.yaml",

        ],
        2: [
            f"python train.py --config configs/config_focal_loss_v1.yaml",
            f"python train.py --config configs/config_focal_loss.yaml",
        ],
        3: [
            f"python train.py --config configs/deeplab_combine.yaml",
            f"python train.py --config configs/deeplab_cross.yaml",
        ],
        4: [
            f"python train.py --config configs/deeplab_dice.yaml",
            f"python train.py --config configs/deeplab_focal.yaml",
        ],
        5: [
            f"python train.py --config configs/unet_cross_v2.yaml",
            f"python train.py --config configs/unet_cross_v3.yaml",
        ],
        6: [
            f"python train.py --config configs/unet_cross.yaml",
            f"python train.py --config configs/unet_resnet_combinedLoss.yaml",
        ],
        7: [
            f"python train.py --config configs/unet_resnet_cross.yaml",
            f"python train.py --config configs/unet_resnet_dice_loss.yaml",
        ],
        8: [
            f"python train.py --config configs/unet_resnet_focal_loss.yaml",
            f"python train.py --config configs/unet_v2_combinedLoss.yaml",
            
        ],
        9: [
            f"python train.py --config configs/unet_v2_cross.yaml",
            f"python train.py --config configs/unet_v2_dice_loss.yaml",
            
        ],
        10: [
            f"python train.py --config configs/unet_v2_focal_loss.yaml",      
        ],
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