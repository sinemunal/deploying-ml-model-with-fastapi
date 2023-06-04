# Example post script
import requests

url = "https://udaicty-project3-fastapi.onrender.com/inference"
request_body = {
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

x = requests.post(url, json=request_body)

print("Response text: ", x.text)
print("Response status: ", x)
