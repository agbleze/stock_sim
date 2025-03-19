import os

def add_to_path(directory):
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Path to the shell configuration file (e.g., .bashrc)
    bashrc_path = os.path.expanduser("~/.bashrc")
    
    # Check if the directory is already in PATH
    with open(bashrc_path, "r") as bashrc:
        lines = bashrc.readlines()
        if any(directory in line for line in lines):
            print(f"Directory '{directory}' is already in PATH.")
            return

    # Add the directory to PATH
    export_command = f'\nexport PATH="$PATH:{directory}"\n'
    with open(bashrc_path, "a") as bashrc:
        bashrc.write(export_command)
        print(f"Added '{directory}' to PATH in {bashrc_path}.")

    # Reload the .bashrc file (optional step for immediate use)
    os.system(f"source {bashrc_path}")

# Example usage
directory_to_add = "/bin/flytectl"
add_to_path(directory_to_add)
