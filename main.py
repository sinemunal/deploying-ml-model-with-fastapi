# FastAPI code for deployment.
import pandas as pd
from fastapi import FastAPI
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


from model_performance import cat_features, _calculate_inference


# Declare the data objects with its components and their type.
class InferenceSetting(BaseModel):
    name: str
    description: Optional[str] = None
    id: int
    data: Dict[str, List[Any]]

    class Config:
        schema_extra = {
            "example": {
                "name": "Inference1",
                "description": "Inference data required for the model",
                "data": {
                    "age": [31],
                    "workclass": ["Self-emp-inc"],
                    "fnlgt": [117963],
                    "education": ["Doctorate"],
                    "education-num": [16],
                    "marital-status": ["Never-married"],
                    "occupation": ["Prof-specialty"],
                    "relationship": ["Own-child"],
                    "race": ["White"],
                    "sex": ["Male"],
                    "capital-gain": [0],
                    "capital-loss": [0],
                    "hours-per-week": [40],
                    "native-country": ["United-States"],
                },
                "id": 1,
            }
        }


class InferenceResult(BaseModel):
    name: str
    description: Optional[str] = None
    result: List[int]

    class Config:
        schema_extra = {
            "example": {
                "name": "InferenceResult1",
                "description": "Inference result",
                "result": [0],
            }
        }


# Save inference settings and results from POST method in the memory.
inference_settings = {}
inference_results = {}

# Initialize FastAPI instance
app = FastAPI()


@app.get("/")
async def greeting():
    return {"This is home to classification model!"}


# Implement POST method to do inference and return the results.
@app.post("/inference/", response_model=InferenceResult)
async def create_inference(inference: InferenceSetting):
    inference_settings[inference.id] = inference

    inference_data = pd.DataFrame(inference.data)

    _, result = _calculate_inference(inference_data, None, cat_features)

    inference_result = {
        "name": f"InferenceResult{inference.id}",
        "description": "Inference result",
        "result": list(result),
    }
    inference_results[inference.id] = inference_result

    return inference_result


# Implement GET that returns the inference and the corresponding data with the given id
@app.get("/inference/{id}")
async def get_inference(id: int):
    try:
        result = inference_results[id]
        setting: InferenceSetting = inference_settings[id]
    except:
        return f"Inference with id: {id} not found."

    return {
        "fetch": f"Fetched inference result: {result} for data file: {setting.data}"
    }
