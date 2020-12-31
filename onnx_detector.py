
import onnxruntime

class ONNXDetector(object):

    def __init__(self,
    model_path,):
        self.session = onnxruntime.InferenceSession(model_path)

    def run(self):
        print(self.session.get_outputs()[0].name)


if __name__ == "__main__":
    detector = ONNXDetector('weights/det_head_d11bd539.onnx')
    detector.run()