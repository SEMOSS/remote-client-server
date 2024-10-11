import os
import datetime
import torch
from typing import Optional
from diffusers import PixArtAlphaPipeline
from io import BytesIO
import base64
import logging
from globals.globals import set_server_status
from model_utils.model_config import get_short_name

logger = logging.getLogger(__name__)


class ImageGen:
    def __init__(
        self,
        model_name: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        short_name = get_short_name()
        # Used in development mode if you are running outside of a docker container otherwise model files should be loaded in attached volume
        model_files_local = os.environ.get("LOCAL_FILES") == "True"
        if model_files_local:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_files_path = os.path.join(
                script_dir, "..", "..", "model_files", short_name
            )
        else:
            model_files_path = f"/app/model_files/{short_name}"
        logger.info(f"Loading the model from path: {model_files_path}")

        try:
            # TODO: Change this to generic pipeline
            self.pipe = PixArtAlphaPipeline.from_pretrained(
                # If you want to download the model files dynamically, use `self.model_name` instead of the model directory below
                model_files_path,
                torch_dtype=(
                    torch.float16
                    if self.device.type == "cuda" or self.device.type == "cuda:0"
                    else torch.float32
                ),
                use_safetensors=True,
            )
            self.pipe.to(self.device)
            logger.info("Image model loaded successfully.")
            set_server_status("READY: ImageGen initialized.")

        except Exception as e:
            logger.exception("Failed to initialize ImageGen: %s", e)
            set_server_status("ImageGen FAILED to initialize.")
            raise

    def generate(
        self,
        prompt: str,
        consistency_decoder: Optional[bool] = False,
        negative_prompt: Optional[str] = None,
        guidance_scale: Optional[float] = 7.5,
        num_inference_steps: Optional[int] = 50,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        seed: Optional[int] = None,
        file_name: Optional[
            str
        ] = "generated_image.png",  # JUST FOR TESTING -- useful to verify images from different clients
        **kwargs,
    ) -> dict:

        logger.info("Generating image...")

        if self.device.type == "cuda":
            logger.info("Using GPU for image generation.")
        else:
            logger.info("Using CPU for image generation.")

        # If using DALL-E 3 Consistency Decoder.. This takes a long time..
        # Varitaional Autoencoder (VAE) enhances performance of image generation pipeline
        # by ensuring high quality coherent outputs
        if consistency_decoder:
            from diffusers import ConsistencyDecoderVAE

            self.pipe.vae = ConsistencyDecoderVAE.from_pretrained(
                "openai/consistency-decoder", torch_dtype=torch.float16
            )
            self.pipe.vae.to(self.device)

        # If seed is not provided by user, generate a random seed.. This is normal seed process
        # else we use the seed provided by the user
        if seed is not None and seed > 0:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)
            seed = generator.seed()

        # Move inputs to the same device as model
        inputs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": int(num_inference_steps),
            "height": int(height),
            "width": int(width),
            "generator": generator,
        }

        self.pipe.enable_attention_slicing()

        start_time = datetime.datetime.now()
        outputs = self.pipe(
            **{
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        )
        end_time = datetime.datetime.now()
        generation_time = (end_time - start_time).total_seconds()

        image = outputs.images[0]

        # Uncomment this for development purposes if you want to verify the image by saving to this directory.
        # unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # image.save(os.path.join("./", f"{file_name}-{unique_identifier}.png"))

        # Converting image to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        base64_str = base64.b64encode(buffered.getvalue()).decode()

        if base64_str is None or base64_str == "":
            base64_str = "There was a problem converting the image to base64."

        response = {
            "generation_time": int(generation_time),
            "seed": str(seed),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": int(num_inference_steps),
            "height": int(height),
            "width": int(width),
            "model_name": self.model_name,
            "vae_model_name": (
                "openai/consistency-decoder" if consistency_decoder else "default"
            ),
            "image": base64_str,
        }

        return response
