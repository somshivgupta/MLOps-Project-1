from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataIngestionArtifact,
    ModelEvaluationArtifact
)
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    production_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
        model_trainer_artifact: ModelTrainerArtifact
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    # ---------- Data preprocessing helpers ----------

    def _map_gender_column(self, df):
        logging.info("Mapping 'Gender' to binary values")
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
        return df

    def _drop_id_column(self, df):
        logging.info("Dropping 'id' column if exists")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def _create_dummy_columns(self, df):
        logging.info("Creating dummy variables")
        return pd.get_dummies(df, drop_first=True)

    def _rename_columns(self, df):
        logging.info("Renaming vehicle age columns and casting types")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    # ---------- MODEL EVALUATION ----------

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            logging.info("Loading test dataset...")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            # Apply same preprocessing used during training
            x = self._map_gender_column(x)
            x = self._drop_id_column(x)
            x = self._create_dummy_columns(x)
            x = self._rename_columns(x)

            # Load trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            trained_f1 = self.model_trainer_artifact.metric_artifact.f1_score

            # Load production model if exists
            production_f1 = None
            if self.model_eval_config.production_model_path is not None:
                try:
                    production_model = load_object(
                        file_path=self.model_eval_config.production_model_path
                    )
                    y_hat_prod = production_model.predict(x)
                    production_f1 = f1_score(y, y_hat_prod)
                except:
                    production_f1 = None  # No production model saved yet

            best_score = 0 if production_f1 is None else production_f1
            is_accepted = trained_f1 > best_score
            diff = trained_f1 - best_score

            return EvaluateModelResponse(
                trained_model_f1_score=trained_f1,
                production_model_f1_score=production_f1,
                is_model_accepted=is_accepted,
                difference=diff
            )

        except Exception as e:
            raise MyException(e, sys)

    # ---------- FINAL ARTIFACT ----------

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation...")

            eval_response = self.evaluate_model()

            artifact = ModelEvaluationArtifact(
                is_model_accepted=eval_response.is_model_accepted,
                changed_accuracy=eval_response.difference,
                production_model_path=self.model_eval_config.production_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path
            )

            logging.info(f"Model Evaluation Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise MyException(e, sys)