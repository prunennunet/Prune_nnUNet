import subprocess


def execute_command(cmd):
    """Execute the command and stream the output to the console."""
    cmd_str = ' '.join(cmd)
    print(f"Executing command: {cmd_str}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
    except FileNotFoundError:
        print(f"Command not found: {cmd[0]}")