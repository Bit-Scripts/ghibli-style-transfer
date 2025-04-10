# reassemble_video.py
import os
import subprocess

# Recomposition vidéo sans audio
cmd_video = (
    "ffmpeg -y -framerate 24 -i styled_frames/frame_%04d.png "
    "-c:v libx264 -pix_fmt yuv420p temp_video.mp4"
)
os.system(cmd_video)

# Récupère l'audio de la vidéo d'origine
cmd_audio = "ffmpeg -y -i video.mp4 -q:a 0 -map a temp_audio.aac"
os.system(cmd_audio)

# Fusionne audio + nouvelle vidéo stylisée
cmd_final = "ffmpeg -y -i temp_video.mp4 -i temp_audio.aac -c:v copy -c:a aac output_stylized.mp4"
os.system(cmd_final)

print("🎬 Vidéo finale générée : output_stylized.mp4")
