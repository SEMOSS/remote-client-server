import logging

import torch

from gaas.model_manager.model_manager import ModelManager

from pydantic_models.request_models import EmotionRequest

from pydantic_models.response_models import EmotionResponse

logger = logging.getLogger(__name__)


class EmotionGen:

    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.device = model_manager.device
        # Initialize the model using ModelManager
        model_files_path = "SamLowe/roberta-base-go_emotions"
        self.model_manager.initialize_emotion_model(model_files_path)
        # Load model and tokenizer from ModelManager
        self.tokenizer = model_manager.tokenizer
        self.model = model_manager.model
        # Load emotion labels from the model config
        self.emotion_labels = (
            self.model.config.id2label
        )  # This ensures we use the correct emotion names

    # def generate(self, request: EmotionRequest) -> dict:
    #     """Generate emotion classification results."""
    #     try:
    #         logger.info("Starting emotion classification...")

    #         # Tokenize input
    #         encoded_input = self.tokenizer(
    #             request.text, return_tensors="pt", truncation=True, padding=True
    #         ).to(self.device)

    #         with torch.no_grad():
    #             # Run the model inference
    #             outputs = self.model(**encoded_input)
    #             scores = torch.sigmoid(outputs.logits).cpu().tolist()[0]
    #         print(scores)
    #         # Convert scores to emotion labels
    #         emotion_scores = [
    #             {"label": self.emotion_labels[i], "confidence": score}
    #             for i, score in enumerate(scores)
    #             if score >= request.confidence_threshold  # Filter by confidence threshold
    #         ]

    #         # Sort and filter top emotions
    #         emotion_scores = sorted(emotion_scores, key=lambda x: x["confidence"], reverse=True)
    #         top_emotions = emotion_scores[: request.top_k] if not request.return_all_scores else emotion_scores

    #         logger.info(f"Detected emotions: {top_emotions}")

    #         # Convert to dictionary before returning
    #         return {
    #             "text": request.text,
    #             "emotions": top_emotions
    #         }

    #     except Exception as e:
    #         logger.error(f"Error during emotion classification: {str(e)}", exc_info=True)
    #         raise RuntimeError("Failed to classify emotions")
    def generate(self, request: EmotionRequest) -> dict:
        """Generate emotion classification results for single or batch inference."""
        try:
            logger.info("Starting emotion classification...")
            # Convert single string input to a list if needed
            texts = request.text if isinstance(request.text, list) else [request.text]
            # Tokenize input
            encoded_inputs = self.tokenizer(
                texts, return_tensors="pt", truncation=True, padding=True
            ).to(self.device)

            with torch.no_grad():
                # Run batch inference
                outputs = self.model(**encoded_inputs)
                scores_batch = torch.sigmoid(outputs.logits).cpu().tolist()
            results = []
            for text, scores in zip(texts, scores_batch):
                emotion_scores = [
                    {"label": self.emotion_labels[i], "confidence": score}
                    for i, score in enumerate(scores)
                    if score >= request.confidence_threshold
                ]
                emotion_scores.sort(key=lambda x: x["confidence"], reverse=True)
                top_emotions = (
                    emotion_scores[: request.top_k]
                    if not request.return_all_scores
                    else emotion_scores
                )

                results.append(EmotionResponse(text=text, emotions=top_emotions).model_dump())
            logger.info("Emotion Classification completed successfully.")

            return {"results": results}

        except Exception as e:

            logger.error(
                f"Error during emotion classification: {str(e)}", exc_info=True
            )

            raise RuntimeError("Failed to classify emotions")
