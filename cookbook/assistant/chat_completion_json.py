from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()
def sdk_objects():
    class ResearchPaperExtraction(BaseModel):
        title: str
        authors: list[str]
        abstract: str
        keywords: list[str]

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure."},
            {"role": "user", "content": '''Personalize Segment Anything Model with One Shot,https://huggingface.co/papers/2305.03048,2023-05-04,"Driven by large-data pre-training, Segment Anything Model (SAM) has been demonstrated as a powerful and promptable framework, revolutionizing the segmentation models. Despite the generality, customizing SAM for specific visual concepts without man-powered prompting is under explored, e.g., automatically segmenting your pet dog in different images. In this paper, we propose a training-free Personalization approach for SAM, termed as PerSAM. Given only a single image with a reference mask, PerSAM first localizes the target concept by a location prior, and segments it within other images or videos via three techniques: target-guided attention, target-semantic prompting, and cascaded post-refinement. In this way, we effectively adapt SAM for private use without any training. To further alleviate the mask ambiguity, we present an efficient one-shot fine-tuning variant, PerSAM-F. Freezing the entire SAM, we introduce two learnable weights for multi-scale masks, only training 2 parameters within 10 seconds for improved performance. To demonstrate our efficacy, we construct a new segmentation dataset, PerSeg, for personalized evaluation, and test our methods on video object segmentation with competitive performance. Besides, our approach can also enhance DreamBooth to personalize Stable Diffusion for text-to-image generation, which discards the background disturbance for better target appearance learning. Code is released at this https URL"'''}
        ],
        response_format=ResearchPaperExtraction,
    )
    print(completion.to_json())
    research_paper = completion.choices[0].message.parsed
    print(research_paper)

def manual_schema():
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
            {"role": "user", "content": "how can I solve 8x + 7 = -23"}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "math_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"}
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False
                            }
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    print(response.to_json())
    parsed = response.choices[0].message.content
    import pdb
    pdb.set_trace()
    import json
    print(json.dumps(json.loads(parsed), indent=4))

{
    "steps": [
        {
            "explanation": "Start with the original equation: 8x + 7 = -23",
            "output": "8x + 7 = -23"
        },
        {
            "explanation": "Subtract 7 from both sides to isolate the term with x: 8x + 7 - 7 = -23 - 7",
            "output": "8x = -30"
        },
        {
            "explanation": "Now, divide both sides by 8 to solve for x: 8x/8 = -30/8",
            "output": "x = -15/4"
        },
        {
            "explanation": "This simplifies to: x = -3.75",
            "output": "x = -3.75"
        }
    ],
    "final_answer": "x = -3.75"
}

if __name__ == '__main__':
    manual_schema()
