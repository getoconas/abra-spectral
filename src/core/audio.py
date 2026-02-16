import librosa
import soundfile as sf
import os

def cargar_audio(ruta, sr=None, mono=False, offset=0.0, duration=None):
  if not os.path.exists(ruta):
    raise FileNotFoundError(f"❌ No encuentro el archivo: {ruta}")
  
  print(f"🏔️ Cargando audio desde: {ruta}...")
  y, sr = librosa.load(ruta, sr=sr, mono=mono, offset=offset, duration=duration)
  return y, sr

def guardar_wav(ruta, audio, sr):
  print(f"💾 Guardando: {ruta}")
  # Transponemos si es estéreo para que soundfile lo entienda
  if audio.ndim > 1 and audio.shape[0] == 2:
    audio = audio.T
  sf.write(ruta, audio, sr, subtype='PCM_24')