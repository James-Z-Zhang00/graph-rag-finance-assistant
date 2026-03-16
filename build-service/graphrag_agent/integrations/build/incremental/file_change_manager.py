import os
import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from graphrag_agent.config.settings import FILE_REGISTRY_PATH

class FileChangeManager:
    """
    Tracks file change state for incremental graph builds.

    Responsibilities:
    1. Scan the file directory and compute SHA256 hashes
    2. Compare against the stored registry to detect added/modified/deleted files
    3. Persist the updated registry to disk
    """

    def __init__(self, files_dir: str, registry_path: str = None):
        """
        Args:
            files_dir: Directory to monitor for changes.
            registry_path: Path to the file registry JSON; defaults to the configured path.
        """
        if registry_path is None:
            registry_path = str(FILE_REGISTRY_PATH)

        self.files_dir = Path(files_dir)
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the file registry from disk.

        Returns:
            Dict mapping relative file paths to their metadata.
        """
        if not self.registry_path.exists():
            return {}

        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            print("Failed to load file registry — a new registry will be created.")
            return {}

    def _save_registry(self):
        """Persist the file registry to disk."""
        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, ensure_ascii=False, indent=2)
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute the SHA256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest string, or empty string on error.
        """
        hash_obj = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"Failed to compute hash for {file_path}: {e}")
            return ""

    def _scan_current_files(self) -> Dict[str, Dict[str, Any]]:
        """
        Scan the monitored directory and collect metadata for all files.

        Returns:
            Dict mapping relative file paths to their metadata.
        """
        current_files = {}

        for root, _, files in os.walk(self.files_dir):
            for filename in files:
                file_path = Path(root) / filename
                rel_path = str(file_path.relative_to(self.files_dir))

                file_hash = self._compute_file_hash(file_path)
                if not file_hash:
                    continue

                file_info = {
                    "hash": file_hash,
                    "size": file_path.stat().st_size,
                    "last_modified": file_path.stat().st_mtime,
                    "last_scanned": time.time()
                }
                current_files[rel_path] = file_info

        return current_files

    def detect_changes(self) -> Dict[str, List[str]]:
        """
        Detect added, modified, and deleted files relative to the registry.

        Returns:
            Dict with keys 'added', 'modified', 'deleted', each mapping to a list of paths.
        """
        current_files = self._scan_current_files()

        added_files = []
        modified_files = []
        deleted_files = []

        for file_path, file_info in current_files.items():
            if file_path not in self.registry:
                added_files.append(file_path)
            elif file_info["hash"] != self.registry[file_path]["hash"]:
                modified_files.append(file_path)

        for file_path in self.registry:
            if file_path not in current_files:
                deleted_files.append(file_path)

        return {
            "added": added_files,
            "modified": modified_files,
            "deleted": deleted_files
        }

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Return the registry metadata for a file.

        Args:
            file_path: Relative file path.

        Returns:
            Metadata dict, or empty dict if not found.
        """
        return self.registry.get(file_path, {})

    def update_registry(self):
        """Re-scan the directory and persist the updated registry."""
        self.registry = self._scan_current_files()
        self._save_registry()
        print(f"File registry updated — {len(self.registry)} files recorded.")

    def update_file_status(self, file_path: str, status: Dict[str, Any]):
        """
        Merge additional status fields into a file's registry entry.

        Args:
            file_path: Relative file path.
            status: Status fields to merge.
        """
        if file_path in self.registry:
            self.registry[file_path].update(status)
            self._save_registry()
    
    def register_file_processing(self, file_path: str, processing_info: Dict[str, Any]):
        """
        Append a processing record to a file's history.

        Args:
            file_path: Relative file path.
            processing_info: Processing details (e.g. duration, node count).
        """
        if file_path in self.registry:
            if "processing_history" not in self.registry[file_path]:
                self.registry[file_path]["processing_history"] = []
            
            processing_record = {
                "timestamp": datetime.now().isoformat(),
                **processing_info
            }
            
            self.registry[file_path]["processing_history"].append(processing_record)
            self.registry[file_path]["last_processed"] = processing_record["timestamp"]
            self._save_registry()