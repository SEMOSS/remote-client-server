import pytest
from gaas.image_gen import ImageGen


@pytest.fixture
def image_gen():
    # TODO: Check if server running dev
    return ImageGen()


def test_image_generation(image_gen):
    prompt = "A beautiful landscape with mountains and a lake"
    negative_prompt = "No people, no buildings"
    guidance_scale = 7.5
    num_inference_steps = 50
    height = 512
    width = 512
    seed = 42

    response, chunks = image_gen.generate_image(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        seed=seed,
    )

    expected_keys = [
        "total_chunks",
        "chunk_size",
        "generation_time",
        "seed",
        "prompt",
        "negative_prompt",
        "guidance_scale",
        "num_inference_steps",
        "height",
        "width",
        "model_name",
        "vae_model_name",
    ]
    for key in expected_keys:
        assert key in response

    assert response["prompt"] == prompt
    assert response["negative_prompt"] == negative_prompt
    assert response["guidance_scale"] == guidance_scale
    assert response["num_inference_steps"] == num_inference_steps
    assert response["height"] == height
    assert response["width"] == width
    assert response["seed"] == str(seed)

    assert response["model_name"] == "PixArt-alpha/PixArt-XL-2-1024-MS"

    assert len(chunks) > 0
    assert response["total_chunks"] == len(chunks)
