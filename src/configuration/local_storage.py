import os
import shutil
import pickle

class LocalStorageService:
    """
    Local Storage Service that mimics the interface of SimpleStorageService (AWS).
    Stores files in a local folder instead of S3.
    """

    def __init__(self, base_dir="local_models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _full_path(self, filename: str):
        return os.path.join(self.base_dir, filename)

    def upload_file(self, from_file, to_filename, bucket_name=None, remove=False):
        """Save file locally (copy)."""
        dest_path = self._full_path(to_filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(from_file, dest_path)
        if remove:
            os.remove(from_file)

    def load_model(self, filename, bucket_name=None):
        """Load pickled model from local storage."""
        path = self._full_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found locally: {path}")

        with open(path, "rb") as f:
            return pickle.load(f)

    def s3_key_path_available(self, bucket_name, s3_key):
        """Check if model exists locally."""
        return os.path.exists(self._full_path(s3_key))
