import tensorflow as tf
import os 
import cv2


def representative_dataset_gen():
    for filename in os.listdir("/home/c_franklin/quant_calib_dataset/coco/"):
        img_path = os.path.join("/home/c_franklin/quant_calib_dataset/coco/", filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                yield_data_dict = {}
                normalized_img = img / 255.0
                yield_data_dict["images"] = normalized_img
                yield yield_data_dict

converter = tf.lite.TFLiteConverter.from_saved_model("../yolo_inference/saved_model",)
# INT8 Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_disable_per_channel = False
converter._experimental_disable_batchmatmul_unfold = not False #True
converter.representative_dataset = representative_dataset_gen
tflite_model = converter.convert()
with open('yolov8n_relu6_300_f.tflite', 'wb') as w:
    w.write(tflite_model)


# Full Integer Quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.SELECT_TF_OPS,
]
converter._experimental_disable_per_channel = False
converter._experimental_disable_batchmatmul_unfold = not False
converter.representative_dataset = representative_dataset_gen

inf_type = tf.int8

converter.inference_input_type = inf_type
converter.inference_output_type = inf_type
tflite_model = converter.convert()
with open('saved_model/output_full_integer_quant.tflite', 'wb') as w:
    w.write(tflite_model)


