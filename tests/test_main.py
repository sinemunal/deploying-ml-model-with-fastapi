import pytest
from fastapi.testclient import TestClient
from main import app


# Instantiate the testing client with our app.
client = TestClient(app)


@pytest.fixture
def inference_setting1():
    return {
        "name": "Inference1",
        "description": "Inference data required for the model",
        "data": {
            "age": [61],
            "workclass": ["Self-emp-inc"],
            "fnlgt": [117963],
            "education": ["Doctorate"],
            "education-num": [16],
            "marital-status": ["Never-married"],
            "occupation": ["Prof-specialty"],
            "relationship": ["Own-child"],
            "race": ["White"],
            "sex": ["Female"],
            "capital-gain": [0],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        },
        "id": 1,
    }


@pytest.fixture
def inference_setting2():
    return {
        "name": "Inference1",
        "description": "Inference data required for the model",
        "data": {
            "age": [41],
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
        "id": 2,
    }


# Test GET method
def test_api_gets_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == ["This is home to classification model!"]


# Test POST method with "/inference" endpoint
def test_create_inference_for_a_data_point(inference_setting1):
    r = client.post(
        "/inference/",
        headers={"X-Token": "coneofsilence"},
        json=inference_setting1,
    )
    assert r.status_code == 200
    assert r.json() == {
        "name": "InferenceResult1",
        "description": "Inference result",
        "result": [0],
    }


def test_create_inference_for_different_data_point(inference_setting2):
    r = client.post(
        "/inference/",
        headers={"X-Token": "coneofsilence"},
        json=inference_setting2,
    )
    assert r.status_code == 200
    assert r.json() == {
        "name": "InferenceResult2",
        "description": "Inference result",
        "result": [0],
    }


# Test GET method with "/inference/1" endpoint
def test_get_inference(inference_setting1):
    r = client.get(
        "/inference/1",
        headers={"X-Token": "coneofsilence"},
    )
    expected_result = {
        "name": "InferenceResult1",
        "description": "Inference result",
        "result": [0],
    }
    expected_data = inference_setting1["data"]

    assert r.status_code == 200
    assert r.json() == {
        "fetch": f"Fetched inference result: {expected_result} for data file: {expected_data}"
    }


def test_get_inference_handles_when_result_is_not_found():
    r = client.get(
        "/inference/3",
        headers={"X-Token": "coneofsilence"},
    )
    assert r.status_code == 200
    assert r.json() == f"Inference with id: 3 not found."


def test_get_malformed():
    r = client.get("/inference")
    assert r.status_code != 200
