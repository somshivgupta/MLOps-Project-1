from src.configuration.local_storage import LocalStorageService
from src.exception import MyException
from src.entity.ss_estimator import MyModel
import sys
from pandas import DataFrame

class Proj1Estimator:
    """
    This class is used to save and retrieve our model from local storage 
    instead of S3 and perform prediction.
    """

    def __init__(self, bucket_name, model_path):
        """
        :param bucket_name: Ignored in local mode (kept for compatibility)
        :param model_path: File path/name for storing model locally
        """
        self.bucket_name = bucket_name      # not used, but kept for compatibility
        self.s3 = LocalStorageService()     # <-- LOCAL BACKEND
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path):
        """Check if model exists locally."""
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=model_path
            )
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """Load model from local storage."""
        try:
            return self.s3.load_model(
                filename=self.model_path,
                bucket_name=self.bucket_name
            )
        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, from_file, remove: bool = False) -> None:
        """Save model to local storage."""
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: DataFrame):
        """Use loaded model to perform prediction."""
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)
