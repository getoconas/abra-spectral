import librosa.display
import matplotlib.pyplot as plt

def graficar_espectrograma(S_db, sr, titulo, ruta_salida):
  plt.figure(figsize=(12, 6))
  librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
  plt.colorbar(format='%+2.0f dB')
  plt.title(titulo)
  plt.tight_layout()
  plt.savefig(ruta_salida)
  print(f"🖼️ Gráfico guardado en: {ruta_salida}")
  plt.close() # Importante cerrar para liberar memoria