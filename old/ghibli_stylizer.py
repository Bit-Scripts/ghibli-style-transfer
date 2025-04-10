import os
import shutil
import cv2
import subprocess

# === CONFIGURATION ===
input_video = "AnimeGANv2/video.mp4"
output_video = "video_ghibli_style.mp4"
fps = 24
style = "Hayao"  # ou "Shinkai", "Paprika"

base_dir = os.path.abspath(".")
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")

# === PRÉPARATION ===
for folder in [input_dir, output_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

# === 1. Extraction des frames ===
print("🎞️ Extraction des frames depuis :", input_video)
cap = cv2.VideoCapture(input_video)
i = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_path = os.path.join(input_dir, f"frame_{i:04d}.png")
    cv2.imwrite(frame_path, frame)
    i += 1
cap.release()
print(f"✅ {i} frames extraites dans '{input_dir}'")

# === 2. Stylisation avec AnimeGANv2 ===
print(f"🎨 Application du style AnimeGANv2 : {style}")
cmd = (
    f"python AnimeGANv2/test.py "
    f"--checkpoint_dir AnimeGANv2/checkpoint/generator_{style}_weight "
    f"--test_dir {input_dir} "
    f"--save_dir {output_dir}"
)
subprocess.run(cmd, shell=True, check=True)

# === 3. Recomposition vidéo sans audio ===
print("🧵 Recomposition de la vidéo stylisée...")
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(fps),
    "-i", f"{output_dir}/frame_%04d.png",
    "-c:v", "libx264", "-pix_fmt", "yuv420p",
    "temp_video.mp4"
])

# === 4. Extraction de l'audio original ===
print("🔊 Extraction de l'audio depuis la vidéo d'origine...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", input_video,
    "-vn", "-acodec", "copy", "temp_audio.aac"
])

# === 5. Fusion audio + vidéo ===
print("🎬 Fusion finale de la vidéo et de l’audio...")
subprocess.run([
    "ffmpeg", "-y",
    "-i", "temp_video.mp4",
    "-i", "temp_audio.aac",
    "-c:v", "copy", "-c:a", "aac",
    output_video
])

print(f"✅ Vidéo stylisée générée : {output_video}")
