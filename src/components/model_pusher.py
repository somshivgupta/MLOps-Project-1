import sys
import os
from src.configuration.local_storage import LocalStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    def __init__(self,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):

        self.local_storage = LocalStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Starting model pusher...")

            # Source: trained model from model trainer output
            src_model_path = self.model_evaluation_artifact.trained_model_path

            # Target: production model location (e.g., saved_models/model.pkl)
            dst_model_path = self.model_pusher_config.production_model_path

            # Ensure destination folder exists
            os.makedirs(os.path.dirname(dst_model_path), exist_ok=True)

            logging.info(f"Saving production model to: {dst_model_path}")

            # Save / copy using LocalStorageService
            self.local_storage.upload_file(
                from_file=src_model_path,
                to_filename=dst_model_path
            )

            # Build artifact
            artifact = ModelPusherArtifact(
                saved_model_path=self.local_storage._full_path(dst_model_path)
            )

            logging.info(f"Model pusher artifact created: {artifact}")
            logging.info("Model pusher completed successfully.")
            return artifact

        except Exception as e:
            raise MyException(e, sys) from e