import onnxruntime
from lightning_ocr.onnx.processor import ImageProcessor
from lightning_ocr.onnx.tokenizer import OnnxTokenizer
import numpy as np
import os


class BASE_ONNX_INFER:
    def __init__(self, model_folder: str, onnx_path: str = "model.onnx"):
        self.model_folder = model_folder
        self.onnx_path = onnx_path
        self.processor = ImageProcessor.load_from_file(
            f"{model_folder}/preprocessor_config.json"
        )
        self.load_tokenizer()
        if os.path.exists(onnx_path) is False:
            if os.path.exists(f"{model_folder}/{onnx_path}") is False:
                raise FileNotFoundError(f"{model_folder}/{onnx_path} does not exist")
            else:
                onnx_path = f"{model_folder}/{onnx_path}"

        self.init_ort_session(onnx_path)

    def load_tokenizer(self):
        self.tokenizer = OnnxTokenizer.load_from_file(
            f"{self.model_folder}/vocab.json",
            f"{self.model_folder}/special_tokens_map.json",
        )

    def init_ort_session(self, onnx_path):
        self.ort_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name

    def forward(self, image):
        ort_inputs = {self.input_name: image}
        ort_outs = self.ort_session.run(None, ort_inputs)
        return ort_outs

    def postprocess(self, outputs):
        raise NotImplementedError

    def predict(self, inputs, with_scores: bool = True, char_split: bool = False):
        x, is_batched = self.processor.preprocess(inputs)
        x = self.forward(np.array(x))
        x, scores = self.postprocess(x)

        if not char_split:
            for i in range(len(x)):
                x[i] = "".join(x[i])
                scores[i] = scores[i].cumprod()[-1]

        if not is_batched:
            x = x[0]
            scores = scores[0]

        if with_scores:
            return x, scores
        else:
            return x


def softmax(X, theta=1.0, axis=None):
    """Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p
