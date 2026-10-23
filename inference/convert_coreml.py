import coremltools as ct
import onnx

model = onnx.load("inference/model.onnx")
mlmodel = ct.converters.onnx.convert(model)
mlmodel.save("app/models/detector.mlmodel")
print("Converted to Core ML")
