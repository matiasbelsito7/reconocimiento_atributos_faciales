"""Cargador de configuraciones."""

from pathlib import Path

import yaml

from facial_attributes.config.schemas import (
    ArchitectureConfig,
    AttributesConfig,
    AugmentationConfig,
    CheckpointConfig,
    DataConfig,
    DatasetConfig,
    DatasetPathsConfig,
    DatasetsConfig,
    EarlyStoppingConfig,
    FaceDetectionConfig,
    FaceExtractionConfig,
    HyperparametersConfig,
    ImageConfig,
    InferenceConfig,
    InferenceModelConfig,
    InferenceOutputConfig,
    InputConfig,
    LoggingConfig,
    MLflowConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    PathsConfig,
    PipelineConfig,
    RegularizationConfig,
    SplitsConfig,
    ThresholdsConfig,
    TrainingConfig,
    ValidationConfig,
)


class ConfigLoader:
    """Cargador de configuraciones desde archivos YAML."""

    def __init__(self, config_dir: str | Path = "config") -> None:
        """Inicializar cargador de configuraciones.

        Args:
            config_dir: Directorio de configuraciones.
        """
        self.config_dir = Path(config_dir)

    def load_pipeline(self) -> PipelineConfig:
        """Cargar configuración del pipeline.

        Returns:
            Configuración del pipeline.
        """
        config_path = self.config_dir / "pipeline.yaml"
        return self._load_pipeline_config(config_path)

    def load_model(self) -> ModelConfig:
        """Cargar configuración del modelo.

        Returns:
            Configuración del modelo.
        """
        config_path = self.config_dir / "model.yaml"
        return self._load_model_config(config_path)

    def load_training(self) -> TrainingConfig:
        """Cargar configuración de entrenamiento.

        Returns:
            Configuración de entrenamiento.
        """
        config_path = self.config_dir / "training.yaml"
        return self._load_training_config(config_path)

    def load_inference(self) -> InferenceConfig:
        """Cargar configuración de inferencia.

        Returns:
            Configuración de inferencia.
        """
        config_path = self.config_dir / "inference.yaml"
        return self._load_inference_config(config_path)

    def load_datasets(self) -> DatasetsConfig:
        """Cargar configuración de datasets.

        Returns:
            Configuración de datasets.
        """
        config_path = self.config_dir / "datasets.yaml"
        return self._load_datasets_config(config_path)

    def load_all(
        self,
    ) -> dict[
        str,
        PipelineConfig
        | ModelConfig
        | TrainingConfig
        | InferenceConfig
        | DatasetsConfig,
    ]:
        """Cargar todas las configuraciones.

        Returns:
            Diccionario con todas las configuraciones.
        """
        return {
            "pipeline": self.load_pipeline(),
            "model": self.load_model(),
            "training": self.load_training(),
            "inference": self.load_inference(),
            "datasets": self.load_datasets(),
        }

    def _load_yaml(self, path: Path) -> dict:
        """Cargar archivo YAML.

        Args:
            path: Ruta al archivo YAML.

        Returns:
            Diccionario con la configuración.
        """
        with open(path) as f:
            return yaml.safe_load(f)

    def _load_pipeline_config(self, path: Path) -> PipelineConfig:
        """Cargar configuración del pipeline.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Configuración del pipeline.
        """
        data = self._load_yaml(path)

        paths_data = data.get("paths", {})
        paths = PathsConfig(**paths_data)

        logging_data = data.get("logging", {})
        logging_config = LoggingConfig(**logging_data)

        return PipelineConfig(
            mode=data.get("mode", "training"),
            paths=paths,
            logging=logging_config,
            seed=data.get("seed", 42),
            device=data.get("device", "auto"),
        )

    def _load_model_config(self, path: Path) -> ModelConfig:
        """Cargar configuración del modelo.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Configuración del modelo.
        """
        data = self._load_yaml(path)

        arch_data = data.get("architecture", {})
        architecture = ArchitectureConfig(**arch_data)

        input_data = data.get("input", {})
        input_config = InputConfig(**input_data)

        output_data = data.get("output", {})
        output_config = OutputConfig(**output_data)

        reg_data = data.get("regularization", {})
        regularization = RegularizationConfig(**reg_data)

        return ModelConfig(
            architecture=architecture,
            input=input_config,
            output=output_config,
            regularization=regularization,
        )

    def _load_training_config(self, path: Path) -> TrainingConfig:
        """Cargar configuración de entrenamiento.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Configuración de entrenamiento.
        """
        data = self._load_yaml(path)

        hp_data = data.get("hyperparameters", {})
        hyperparameters = HyperparametersConfig(**hp_data)

        es_data = data.get("early_stopping", {})
        early_stopping = EarlyStoppingConfig(**es_data)

        ckpt_data = data.get("checkpoint", {})
        checkpoint = CheckpointConfig(**ckpt_data)

        data_config_data = data.get("data", {})
        data_config = DataConfig(**data_config_data)

        aug_data = data.get("augmentation", {})
        augmentation = AugmentationConfig(**aug_data)

        logging_data = data.get("logging", {})
        logging_config = LoggingConfig(**logging_data)

        mlflow_data = data.get("mlflow", {})
        mlflow_config = MLflowConfig(**mlflow_data)

        return TrainingConfig(
            seed=data.get("seed", 42),
            hyperparameters=hyperparameters,
            early_stopping=early_stopping,
            checkpoint=checkpoint,
            data=data_config,
            augmentation=augmentation,
            logging=logging_config,
            mlflow=mlflow_config,
        )

    def _load_inference_config(self, path: Path) -> InferenceConfig:
        """Cargar configuración de inferencia.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Configuración de inferencia.
        """
        data = self._load_yaml(path)

        thresholds_data = data.get("thresholds", {})
        thresholds = ThresholdsConfig(**thresholds_data)

        fd_data = data.get("face_detection", {})
        face_detection = FaceDetectionConfig(**fd_data)

        fe_data = data.get("face_extraction", {})
        face_extraction = FaceExtractionConfig(**fe_data)

        opt_data = data.get("optimization", {})
        optimization = OptimizationConfig(**opt_data)

        out_data = data.get("output", {})
        output = InferenceOutputConfig(**out_data)

        inf_data = data.get("inference", {})
        inference_model = InferenceModelConfig(**inf_data)

        return InferenceConfig(
            thresholds=thresholds,
            face_detection=face_detection,
            face_extraction=face_extraction,
            optimization=optimization,
            output=output,
            inference=inference_model,
        )

    def _load_datasets_config(self, path: Path) -> DatasetsConfig:
        """Cargar configuración de datasets.

        Args:
            path: Ruta al archivo de configuración.

        Returns:
            Configuración de datasets.
        """
        data = self._load_yaml(path)

        celeba_data = data.get("celeba", {})
        celeba_config = self._load_dataset_config(celeba_data)

        sample_data = data.get("sample", {})
        sample_config = self._load_dataset_config(sample_data)

        return DatasetsConfig(celeba=celeba_config, sample=sample_config)

    def _load_dataset_config(self, data: dict) -> DatasetConfig:
        """Cargar configuración de un dataset.

        Args:
            data: Diccionario con la configuración.

        Returns:
            Configuración del dataset.
        """
        paths_data = data.get("paths", {})
        paths = DatasetPathsConfig(**paths_data)

        images_data = data.get("images", {})
        images = ImageConfig(**images_data)

        attrs_data = data.get("attributes", {})
        attributes = AttributesConfig(**attrs_data)

        splits_data = data.get("splits", {})
        splits = SplitsConfig(**splits_data)

        validation_data = data.get("validation", {})
        validation = ValidationConfig(**validation_data)

        return DatasetConfig(
            name=data.get("name", ""),
            description=data.get("description", ""),
            source=data.get("source", ""),
            kaggle_dataset=data.get("kaggle_dataset", ""),
            version=data.get("version", "1.0"),
            paths=paths,
            images=images,
            attributes=attributes,
            splits=splits,
            validation=validation,
        )
