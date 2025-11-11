"""
Mixins for extractor functionality.
"""

from typing import Optional
import dtlpy as dl
import logging

logger = logging.getLogger("rag-preprocessor")


class DataloopModelMixin:
    """
    Mixin for extractors that use Dataloop models.
    Provides model lookup, deployment verification, and execution capabilities.
    """

    def __init__(self, model_id: Optional[str] = None, *args, **kwargs):
        """
        Initialize with a model ID.

        Args:
            model_id (Optional[str]): Dataloop model ID
        """
        super().__init__(*args, **kwargs)
        self.model_id = model_id
        self.model = None

        if not model_id:
            logger.info("No model_id provided - will use fallback methods")

    def _get_model(self) -> Optional[dl.Model]:
        """
        Fetch and cache the Dataloop model.

        Returns:
            Optional[dl.Model]: The Dataloop model, or None if not found
        """
        if self.model is None and self.model_id:
            try:
                self.model = dl.models.get(model_id=self.model_id)
                logger.info(f"Loaded model | model_id={self.model_id} name={self.model.name}")
            except Exception as e:
                logger.error(f"Failed to load model | model_id={self.model_id} error={str(e)}")
                return None
        return self.model

    def _check_model_deployed(self, auto_deploy: bool = True):
        """
        Check if the model is deployed and ready.

        Args:
            auto_deploy (bool): If True, automatically deploys the model if not deployed (default: True)

        Returns:
            tuple: (model, status) if model is deployed (or deployment triggered)

        Raises:
            ValueError: If model is not deployed and auto_deploy is False
        """
        model = self._get_model()
        if model is None:
            raise ValueError(f"Could not load model with id: {self.model_id}")

        if model.status != dl.ModelStatus.DEPLOYED:
            if auto_deploy:
                logger.info(f"Model not deployed, deploying now | model_id={self.model_id} status={model.status}")
                try:
                    model.deploy()
                    logger.info(f"Model deployment triggered | model_id={self.model_id}")
                except Exception as e:
                    logger.error(f"Failed to deploy model | model_id={self.model_id} error={str(e)}")
                    raise ValueError(f"Failed to deploy model {self.model_id}: {str(e)}")
            else:
                raise ValueError(
                    f"Model {self.model_id} is not deployed (status: {model.status}). "
                    f"Please deploy the model before using it for predictions."
                )
        else:
            logger.info(f"Model is deployed | model_id={self.model_id} status={model.status}")

        return model, model.status

    def execute_model(self, item: dl.Item, **kwargs) -> dl.Item:
        """
        Execute the model on the item and wait for completion.

        Args:
            item (dl.Item): Dataloop item to process
            **kwargs: Additional parameters for the model

        Returns:
            dl.Item: Updated item after model execution

        Raises:
            ValueError: If model_id is not configured or model not deployed
            Exception: If execution fails or times out
        """
        if not self.model_id:
            raise ValueError("model_id must be provided to execute_model")

        # Check if model is deployed
        self._check_model_deployed()

        model = self._get_model()

        logger.info(f"Executing model | model_id={self.model_id} item_id={item.id}")

        try:
            # Execute prediction
            execution = model.predict(item_ids=[item.id], **kwargs)

            # Wait for execution to complete
            logger.info(f"Waiting for execution to complete | execution_id={execution.id}")
            execution.wait()
            if execution.latest_status['status'] == dl.ExecutionStatus.FAILED:
                raise Exception(
                    f"Model execution failed for item {item.id}: {execution.latest_status.get('message', 'Unknown error')}"
                )
            elif execution.latest_status['status'] == dl.ExecutionStatus.SUCCESS:
                logger.info(f"Model execution successful | execution_id={execution.id}")

            # Refresh item to get updated data
            logger.info(f"Fetching updated item | item_id={item.id}")
            updated_item = dl.items.get(item_id=item.id)

            return updated_item

        except Exception as e:
            logger.error(f"Model execution failed: {str(e)}")
            raise Exception(f"Model execution failed for item {item.id}: {str(e)}")

    def has_dataloop_backend(self) -> bool:
        """
        Check if this extractor is configured to use Dataloop models.

        Returns:
            bool: True if model_id is configured
        """
        return bool(self.model_id)
