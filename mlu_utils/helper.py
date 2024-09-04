import matplotlib.pyplot as plt
import numpy as np
import json


def query(model_predictor, text):
    """Query the model predictor."""

    encoded_text = json.dumps(text).encode("utf-8")

    query_response = model_predictor.predict(
        encoded_text,
        {
            "ContentType": "application/json",
            "Accept": "application/json",
        },
    )
    return query_response


def parse_response(query_response):
    """Parse response and return generated image and the prompt"""

    response_dict = json.loads(query_response)
    return response_dict["generated_images"]


def display_img_and_prompt(img, prmpt):
    """Display hallucinated image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(np.array(img))
    plt.axis("off")
    plt.title(prmpt["prompt"])
    plt.show()
    
def query_response(predictor, prompt):
    
    response = query(predictor, prompt)
    
    for im in parse_response(response):
        display_img_and_prompt(im, prompt)
    return