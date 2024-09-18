from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import logging
# from utils import save_jsonl
import json
from typing import TypeGuard, Any

client = OpenAI()


class ResearchPaper(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    research_area: str = Field(
        ..., description="General field of Deep Learning, e.g., NLP, CV, RL, Audio, Robotics")
    task: str = Field(..., description="Specific task within the research field, e.g., 3D Reconstruction, Imitation Learning")
    contribution_type: str = Field(
        ..., description="Nature of the research contribution, e.g., Model Architecture, Algorithm, Benchmark, Theoretical Analysis, Frameworks, Survey, Rethinking, Technical Report")
    model_type: str = Field(
        ..., description="If applicable, the type of model used in the research, e.g., Transformer, YOLO, GAN")
    dataset: str = Field(
        ..., description="The dataset used or introduced in the research, e.g., MNIST, ImageNet")
    # = Field(..., description="Important terms or concepts related to the paper, e.g., open-vocabulary, real-time, PLMs, multilingual")
    keywords: list[str]

def is_dict(obj: object) -> TypeGuard[dict[object, object]]:
    return isinstance(obj, dict)
def is_list(obj: object) -> TypeGuard[list[object]]:
    return isinstance(obj, list)
def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    i = 0
    for _ in obj.keys():
        i += 1
        if i > n:
            return True
    return False
def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")
    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(value), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value
    return resolved

def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    See https://github.com/openai/openai-python/pull/1655/files
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(def_schema, path=(*path, "$defs", def_name), root=root)

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(definition_schema, path=(*path, "definitions", definition_name), root=root)

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(prop_schema, path=(*path, "properties", key), root=root)
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(items, path=(*path, "items"), root=root)

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(variant, path=(*path, "anyOf", str(i)), root=root)
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(_ensure_strict_json_schema(all_of[0], path=(*path, "allOf", "0"), root=root))
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(entry, path=(*path, "allOf", str(i)), root=root)
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", False) is None:
        json_schema.pop("default")

    # # we can't use `$ref`s if there are also other properties defined, e.g.
    # # `{"$ref": "...", "description": "my description"}`
    # #
    # # so we unravel the ref
    # # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref:
        raise ValueError("schema has ref.")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}")

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")

    return json_schema



def batch_classify(papers_csv, batch_jsonl):
    NEW_COLUMNS = ['research_area', 'task', 'contribution_type', 'model_type', 'dataset', 'keywords']
    instructions = """You are a AI expert specializing in classifying AI research papers. Your role is to perform multiple-label classification of AI research papers. When provided with a paper's title and abstract, it will analyze the content and classify the paper according to a specified JSON schema. The schema will include various fields that define the categories or types the paper belongs to. You are responsible for understanding the research context, identifying relevant keywords, concepts, and themes, and accurately filling out the fields in the JSON schema. You should aim to be precise and efficient in its classification, ensuring that all necessary information is correctly inputted into the fields. When filling out, values must be general and accurate, and any field that cannot be filled should be filled as "None" instead of providing random or inappropriate data. The tone should be professional, clear, and supportive, guiding the user through the process of paper classification in an organized manner."""

    # Add new columns into df.
    df = pd.read_csv(papers_csv)
    for col in NEW_COLUMNS:
        df[col] = None

    reqs = []
    for index, row in df.iterrows():
        prompt = f'TITLE: {row["title"]}\nABSTRACT: {row["abstract"]}'
        prompt = f'title: PyTorch: An Imperative Style, High-Performance Deep Learning Library\nABSTRACT: Deep learning frameworks have often focused on either usability or speed, but not both. PyTorch is a machine learning library that shows that these two goals are in fact compatible: it provides an imperative and Pythonic programming style that supports code as a model, makes debugging easy and is consistent with other popular scientific computing libraries, while remaining efficient and supporting hardware accelerators such as GPUs.'
        schema = _ensure_strict_json_schema(ResearchPaper.model_json_schema(), path=(), root=ResearchPaper.model_json_schema())
        print(json.dumps(schema, indent=4))
        response_format = {
                "type": "json_schema",
                "json_schema": {
                    "schema": schema,
                    "name": ResearchPaper.__name__,
                    'strict': True,
                },
            }
        # print(json.dumps(response_format, indent=4))
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ],
            response_format=ResearchPaper
            # response_format=response_format
        )
        print(completion.to_json())
        research_paper = completion.choices[0].message.parsed
        print(research_paper)

        


        # reqs.append({
        #     'custom_id': f'request-{index}',
        #     'method': 'POST',
        #     'url': 'https://api.openai.com/v1/chat/completions',
        #     'body': {
        #         'messages': [
        #             {"role": "system", "content": instructions},
        #             {"role": "user", "content": prompt}
        #         ],
        #         'response_format: {
        #             ''type: 'json_schema',
        #             'json_schema': {
        #                 'schema': ResearchPaper.model_json_schema(),
        #                 'name': ResearchPaper.__name__
        #             }
      #          "strict": True,
        #         }
        #     }
        # })
        break
    # save_jsonl(batch_jsonl, reqs, 'a')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] - %(message)s')
    batch_classify('./rag/data/papers/combined.csv', './rag/batchinput.jsonl')
