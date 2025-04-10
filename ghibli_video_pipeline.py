import os
import sys
import signal
import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import argparse
import shutil
import matplotlib.pyplot as plt
import time

def handle_sigint(sig, frame):
    print("\n⛔ Interruption manuelle détectée (Ctrl+C).")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_sigint)

# === 1. EXTRACTION DES FRAMES ===
def extract_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), frame)
        i += 1
    cap.release()
    print(f"✅ {i} frames extraites dans '{output_dir}'")

def stylize_frames_tf(model_path, input_dir, output_dir, preview=False, debug_color=False, restore_resolution=True):
    import tensorflow.compat.v1 as tf
    tf.disable_eager_execution()

    os.makedirs(output_dir, exist_ok=True)

    print("🧠 Chargement du modèle TensorFlow (.pb)...")
    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    input_tensor = graph.get_tensor_by_name("input:0")
    output_tensor = graph.get_tensor_by_name("generator/G_MODEL/out_layer/Tanh:0")

    frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not frames:
        print("⚠️ Aucune frame trouvée.")
        return

    with tf.Session(graph=graph) as sess:
        for idx, frame in enumerate(tqdm(frames, desc="✨ Stylisation", unit="image")):
            img_path = os.path.join(input_dir, frame)
            img = cv2.imread(img_path)
            original_shape = img.shape[:2][::-1]

            img_resized = cv2.resize(img, (256, 256))
            img_input = img_resized.astype(np.float32)[None, :, :, ::-1] / 127.5 - 1.0  # BGR->RGB + normalisation

            output = sess.run(output_tensor, feed_dict={input_tensor: img_input})[0]
            output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            if restore_resolution:
                output = cv2.resize(output, original_shape)

            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, frame), output_bgr)

            if (preview or debug_color) and idx == 0:
                plt.imshow(output)
                plt.title("Preview")
                plt.axis('off')
                plt.show()

    print("✅ Stylisation TensorFlow terminée.")

def stylize_frames_onnx(model_path, input_dir, output_dir, preview=False, debug_color=False, restore_resolution=True):
    import onnxruntime as ort

    os.makedirs(output_dir, exist_ok=True)

    print("🧠 Chargement du modèle ONNX...")
    available = ort.get_available_providers()
    print(f"✅ ONNXRuntime providers disponibles : {available}")
    session = ort.InferenceSession(model_path, providers=available)

    providers = ort.get_available_providers()
    print("✅ ONNXRuntime providers disponibles :", providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("Entrées ONNX :", [i.name for i in session.get_inputs()])
    print("Sorties ONNX :", [o.name for o in session.get_outputs()])

    frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".png")])
    if not frames:
        print("⚠️ Aucune frame trouvée.")
        sys.exit(1)

    try:
        for idx, frame in enumerate(tqdm(frames, desc="✨ Stylisation", unit="image")):
            img_path = os.path.join(input_dir, frame)
            img = cv2.imread(img_path)
            original_shape = img.shape[:2][::-1]  # (width, height)

            # Resize pour le modèle (256x256 par défaut)
            img_resized = cv2.resize(img, (256, 256))
            img_input = img_resized.astype(np.float32)[None, :, :, ::-1] / 127.5 - 1.0  # BGR -> RGB + normalisation


            output = session.run(None, {input_name: img_input})[0]
            output = output[0]
            if output.ndim == 3 and output.shape[0] == 3:
                output = output.transpose(1, 2, 0)
            elif output.ndim == 2:
                output = np.stack([output] * 3, axis=-1)  # force RGB pour les images N&B
            output = ((output + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

            # Restauration de la résolution d’origine si demandé
            if restore_resolution:
                output = cv2.resize(output, original_shape)

            print(f"[DEBUG] output.shape = {output.shape}, dtype = {output.dtype}")
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, frame), output_bgr)


            if (preview or debug_color) and idx == 0:
                import matplotlib.pyplot as plt

                if preview:
                    plt.imshow(output)
                    plt.title("Preview")
                    plt.axis('off')
                    plt.show()

                if debug_color:
                    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

                    axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    axs[0, 0].set_title("Image Originale")
                    axs[0, 0].axis('off')

                    axs[0, 1].imshow(output)
                    axs[0, 1].set_title("Image Stylisée")
                    axs[0, 1].axis('off')

                    for i, c in enumerate(['b', 'g', 'r']):
                        axs[1, 0].plot(cv2.calcHist([img], [i], None, [256], [0, 256]), color=c)
                        axs[1, 1].plot(cv2.calcHist([output], [i], None, [256], [0, 256]), color=c)

                    axs[1, 0].set_title("Histogramme Original")
                    axs[1, 1].set_title("Histogramme Stylisé")
                    for ax in axs[1]:
                        ax.set_xlim([0, 256])

                    plt.tight_layout()
                    plt.show()

    except KeyboardInterrupt:
        print("\n⛔ Interruption détectée. Le script s'est arrêté proprement.")

    print("✅ Toutes les images ont été stylisées.")

# === 4. REASSEMBLAGE VIDÉO ===
def reassemble_video(fps, original_video, styled_dir):
    print("🎞️ Recomposition de la vidéo avec ffmpeg...")
    os.system(f"ffmpeg -y -framerate {fps} -i {styled_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p temp_video.mp4")
    os.system(f"ffmpeg -y -i {original_video} -q:a 0 -map a temp_audio.aac")
    os.system("ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -c:v copy -c:a aac output_stylized.mp4")
    print("🎬 Vidéo finale générée : output_stylized.mp4")

# === 4. REASSEMBLAGE VIDÉO ===
def reassemble_video(fps, original_video, styled_dir):
    print("🎞️ Recomposition de la vidéo avec ffmpeg...")
    os.system(f"ffmpeg -y -framerate {fps} -i {styled_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p temp_video.mp4")
    os.system(f"ffmpeg -y -i {original_video} -q:a 0 -map a temp_audio.aac")
    os.system("ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -c:v copy -c:a aac output_stylized.mp4")
    print("🎬 Vidéo finale générée : output_stylized.mp4")

def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return int(round(fps))

def main():
    print("🌸 Bienvenue dans Ghibli Stylizer ! 🎥 -> 🎨")
    # === CONFIGURATION ===
    parser = argparse.ArgumentParser(description="AnimeGANv2 Ghibli Stylizer Pipeline")
    parser.add_argument("--video", type=str, default="video.mp4", help="Chemin vers la vidéo source")
    # parser.add_argument("--model", type=str, default="AnimeGANv2/pb_and_onnx_model/Shinkai_53.onnx", help="Chemin du modèle ONNX")
    parser.add_argument("--model", type=str, default="AnimeGANv2/pb_and_onnx_model/Hayao.onnx", help="Chemin du modèle ONNX")
    parser.add_argument("--preview", action="store_true", help="Affiche un aperçu de la première frame stylisée")
    parser.add_argument("--skip-extract", action="store_true", help="Ne pas extraire les frames si elles existent déjà")
    parser.add_argument("--clean-frames", action="store_true", help="Supprime les anciens frames")
    parser.add_argument("--fps", type=int, default=None, help="Framerate pour la vidéo de sortie (défaut: même que l'entrée)")
    parser.add_argument("--debug-color", action="store_true", help="Affiche un comparatif visuel et les histogrammes de la première frame")
    args = parser.parse_args()

    # === 1. EXTRACTION DES FRAMES ===
    if args.clean_frames:
        print("🧹 Nettoyage des anciens frames...")
        for folder in ["frames", "styled_frames"]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                print(f"🗑️ Dossier supprimé : {folder}")

    if not args.skip_extract:
        extract_frames(args.video, "frames")

    if args.fps is None:
        args.fps = get_video_fps(args.video)
        print(f"🎯 FPS détecté automatiquement : {args.fps}")

    # === 2. CHARGEMENT DU MODÈLE ONNX ===
    # === 3. STYLISATION DES FRAMES ===

    start_time = time.time()
    if args.model.endswith(".onnx"):
        import onnxruntime as ort
        stylize_frames_onnx(args.model, "frames", "styled_frames", args.preview, args.debug_color)
    else:
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        stylize_frames_tf(args.model, "frames", "styled_frames", args.preview, args.debug_color)
    print(f"🕒 Stylisation terminée en {time.time() - start_time:.2f} secondes.")
    print("🖼️ Stylisation terminée.")
    reassemble_video(args.fps, args.video, "styled_frames")
    print("🎬 Vidéo finale créée : output_stylized.mp4")
    print("✨ Pipeline complet exécuté avec succès. À bientôt dans le monde de Ghibli ! 🍃")

if __name__ == "__main__":
    if not shutil.which("ffmpeg"):
        print("❌ ffmpeg n'est pas installé ou pas dans le PATH.")
        sys.exit(1)
    main()
