"""Export the trained Keras model to ONNX for portable/optimized inference."""
from pathlib import Path
import tensorflow as tf
import tf2onnx

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "models" / "ResNet_history.keras"
target = ROOT / "models" / "model.onnx"
model = tf.keras.models.load_model(source, compile=False)
signature = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="image"),)
tf2onnx.convert.from_keras(model, input_signature=signature, opset=17, output_path=str(target))
print(f"Exported {target} ({target.stat().st_size:,} bytes)")
