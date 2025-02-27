import logging
import time
import uuid
from typing import Dict, List
from gaas.model_manager.model_manager import ModelManager
from pydantic_models.request_models import SentimentGenRequest
from pydantic_models.response_models import SentimentGenResponse

logger = logging.getLogger(__name__)


class SentimentGen:
    def __init__(
        self,
        model_manager: ModelManager,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_manager = model_manager
        self.model = model_manager.model
        self.device = model_manager.device
        self.pipelines = model_manager.pipe

    def _create_final_response(self, sentiment_gen_summary, request) -> Dict:
        """Create a properly formatted final response"""
        return {
            "id": f"sentimentgen-{uuid.uuid4()}",
            "object": "sentiment-generation",
            "created": int(time.time()),
            "model": request.model,
            "sentiment_gen_report": sentiment_gen_summary,
        }

    def _format_prediction_result(self, texts, predicted_results) -> List:
        """Formatting the prediction result properly"""
        try:
            formatting_sentiment_gen = []
            for text, prediction in zip(texts, predicted_results):
                each_set = {}
                each_set["text"] = text
                each_set["sentiment_label"] = prediction["label"]
                each_set["sentiment_score"] = prediction["score"]
                formatting_sentiment_gen.append(each_set)

            return formatting_sentiment_gen
        except Exception as e:
            logger.error(f"Error during formatting sentiment prediction: {str(e)}")
            raise

    def _decoding_labels(self, encoded_predicted_results):
        """Decoding the labels"""
        try:
            label_coder = {
                # Multi-class Codes
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL",
                "LABEL_2": "POSITIVE",
            }

            decoded_predicted_results = [
                (
                    {**each, "label": label_coder[each["label"]]}
                    if each.get("label") in label_coder.keys()
                    else each
                )
                for each in encoded_predicted_results
            ]

            return decoded_predicted_results
        except Exception as e:
            logger.error(f"Error during decoding sentiment prediction: {str(e)}")
            raise

    def generate(self, request: SentimentGenRequest) -> SentimentGenResponse:
        """Perform sentiment classification"""
        try:
            logger.info("Starting sentiment generation...")
            texts = request.text

            if isinstance(texts, str):
                texts = [texts]

            # Predict the sentiment and their probability score
            encoded_predicted_results = self.model(texts)
            logger.info(f"Before Decoding - {encoded_predicted_results}")

            decoded_predicted_results = self._decoding_labels(encoded_predicted_results)
            logger.info(f"After Decoding - {decoded_predicted_results}")

            formatted_result = self._format_prediction_result(
                texts, decoded_predicted_results
            )
            logger.info("Completing sentiment generation...")

            return self._create_final_response(formatted_result, request)
        except Exception as e:
            logger.error(f"Error during sentiment generation: {str(e)}")
            raise
