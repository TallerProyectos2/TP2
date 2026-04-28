from __future__ import annotations

import unittest

import numpy as np

from roboflow_runtime import InferenceConfig, infer_one_frame


class FakeModelClient:
    def __init__(self) -> None:
        self.image = None
        self.model_id = None

    def infer(self, image, *, model_id):
        self.image = image
        self.model_id = model_id
        return {"predictions": []}


class FakeWorkflowClient:
    def __init__(self) -> None:
        self.kwargs = None

    def run_workflow(self, **kwargs):
        self.kwargs = kwargs
        return [{"predictions": []}]


def config(target: str) -> InferenceConfig:
    return InferenceConfig(
        mode="local",
        target=target,
        local_api_url="http://127.0.0.1:9001",
        cloud_workflow_api_url="https://serverless.roboflow.com",
        cloud_model_api_url="https://detect.roboflow.com",
        api_key="",
        workspace="workspace",
        workflow="workflow",
        model_id="project/1",
    )


class InferOneFrameTest(unittest.TestCase):
    def test_model_inference_uses_numpy_frame_without_path(self):
        client = FakeModelClient()
        frame = np.zeros((10, 20, 3), dtype=np.uint8)

        infer_one_frame(client, frame, config("model"))

        self.assertIsInstance(client.image, np.ndarray)
        self.assertEqual(client.image.shape, (10, 20, 3))
        self.assertEqual(client.model_id, "project/1")

    def test_workflow_inference_uses_numpy_frame_without_path(self):
        client = FakeWorkflowClient()
        frame = np.zeros((10, 20, 3), dtype=np.uint8)

        infer_one_frame(client, frame, config("workflow"))

        self.assertIsInstance(client.kwargs["images"]["image"], np.ndarray)
        self.assertFalse(client.kwargs["use_cache"])


if __name__ == "__main__":
    unittest.main()
