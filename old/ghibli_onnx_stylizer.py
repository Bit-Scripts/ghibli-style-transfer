# ghibli_onnx_stylizer.py
import os
import onnxruntime as ort
import cv2
import numpy as np
from tqdm import tqdm
import signal
import sys

def handle_sigint(signal, frame):
    print("\n⛔ Interruption manuelle détectée (Ctrl+C).")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# === CONFIGURATION ===
model_path = "AnimeGANv2/pb_and_onnx_model/Shinkai_53.onnx"
input_dir = "frames"
output_dir = "styled_frames"
size = (256, 256)

os.makedirs(output_dir, exist_ok=True)

# === INITIALISATION DE ONNX AVEC ROCm ===
session = ort.InferenceSession(model_path, providers=["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"])
print("Entrées ONNX :", [i.name for i in session.get_inputs()])
print("Sorties ONNX :", [o.name for o in session.get_outputs()])

# === TRAITEMENT DES IMAGES AVEC BARRE DE PROGRESSION ===
frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
if not frames:
    print("⚠️ Aucun fichier .png trouvé dans le dossier 'frames'.")
    sys.exit(1)
try:
    for frame in tqdm(frames, desc="✨ Stylisation", unit="image"):
        img_path = os.path.join(input_dir, frame)
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)

        # Prétraitement
        img_input = img.astype(np.float32)[None, :, :, ::-1] / 127.5 - 1.0

        # Inférence ONNX
        output = session.run(None, {"generator_input:0": img_input})[0]

        # Post-traitement auto-détecté
        output = output[0].transpose(1, 2, 0) if output.shape[1] == 3 else output[0]
        output = ((output + 1.0) * 127.5).astype(np.uint8)

        # Sauvegarde
        cv2.imwrite(os.path.join(output_dir, frame), output)
except KeyboardInterrupt:
    print("\n⛔ Interruption détectée. Le script s'est arrêté proprement.")

print("✅ Toutes les images ont été stylisées avec succès.")
