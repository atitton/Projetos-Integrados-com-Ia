"""Gera um wav por linha de trechos.txt. Uso: python gerar_trechos.py <voz>"""
import sys
from pathlib import Path
import soundfile as sf
from kokoro_onnx import Kokoro

D = Path(__file__).resolve().parent
voz = sys.argv[1] if len(sys.argv) > 1 else "pm_santa"
k = Kokoro(str(D.parent / "models/kokoro-v1.0.onnx"), str(D.parent / "models/voices-v1.0.bin"))
for i, linha in enumerate(D.joinpath("trechos.txt").read_text(encoding="utf-8").splitlines(), 1):
    if not linha.strip():
        continue
    audio, sr = k.create(linha.strip(), voice=voz, speed=1.0, lang="pt-br")
    sf.write(D / f"trechos/{i:02d}.wav", audio, sr)
    print(f"{i:02d} {len(audio) / sr:.2f}")
