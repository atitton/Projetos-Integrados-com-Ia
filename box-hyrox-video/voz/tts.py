"""Gera um wav com Kokoro (kokoro-onnx). Uso: python tts.py <voz> <texto> <saida.wav>"""
import sys
from pathlib import Path
import soundfile as sf
from kokoro_onnx import Kokoro

M = Path(__file__).resolve().parent.parent / "models"
voz, texto, saida = sys.argv[1], sys.argv[2], sys.argv[3]
if Path(texto).is_file():
    texto = Path(texto).read_text(encoding="utf-8")
k = Kokoro(str(M / "kokoro-v1.0.onnx"), str(M / "voices-v1.0.bin"))
audio, sr = k.create(texto.strip(), voice=voz, speed=1.0, lang="pt-br")
sf.write(saida, audio, sr)
print(saida, round(len(audio) / sr, 2))
