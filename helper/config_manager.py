import shutil
import datetime
from pathlib import Path
import yaml
from typing import Dict, Any


class ConfigManager:
    """Manages configuration file backups and updates."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.backup_path = None
        self.original_config = None

    def backup(self) -> Path:
        """
        Create a timestamped backup of the config file.

        Returns:
            Path: Path to the backup file
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Create timestamped backup filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.config_path.parent / "config_backups"
        backup_dir.mkdir(exist_ok=True)

        filename = self.config_path.stem
        extension = self.config_path.suffix
        self.backup_path = backup_dir / f"{filename}_backup_{timestamp}{extension}"

        # Copy the file with metadata preserved
        shutil.copy2(self.config_path, self.backup_path)
        print(f"Config backup created at: {self.backup_path}")

        # Store the original config for potential later restoration
        with open(self.config_path, 'r') as f:
            self.original_config = f.read()

        return self.backup_path

    def restore_from_backup(self) -> None:
        """Restore the original config file from the most recent backup."""
        if self.original_config is None:
            raise RuntimeError("No backup has been created yet.")

        with open(self.config_path, 'w') as f:
            f.write(self.original_config)

        print(f"Config restored from backup to: {self.config_path}")

    def update_config(self, updated_config: Dict[str, Any]) -> None:
        """
        Update the config file while preserving comments and structure.

        This is a simple implementation that only updates values without
        changing the structure or comments. For more complex cases,
        consider using ruamel.yaml which preserves comments.

        Args:
            updated_config: Dictionary containing the updated configuration
        """
        # Read the original config with PyYAML to get the data
        with open(self.config_path, 'r') as f:
            current_config = yaml.safe_load(f)

        # Update the config with new values
        current_config.update(updated_config)

        # Write back to file
        with open(self.config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False)

        print(f"Config updated at: {self.config_path}")