import os
# Importamos NUESTROS módulos
from src.core import audio, dsp
from src.utils import plots

def ejecutar_abra_spectral(nombre_cancion):
  # --- PASO 0: Extraer el nombre base para no sobrescribir ---
  # Esto quita la extensión (ej: "03 - Gorilla.mp3" -> "03 - Gorilla")
  nombre_base = os.path.splitext(nombre_cancion)[0]

  # Definir rutas relativas
  ruta_entrada = os.path.join("data", "input", nombre_cancion)
  carpeta_salida = os.path.join("data", "output")
  
  # Asegurar que existe carpeta de salida
  os.makedirs(carpeta_salida, exist_ok=True)

  # 1. CARGA
  y, sr = audio.cargar_audio(ruta_entrada, duration=None)
  print(f"📊 Audio cargado: {sr}Hz")

  # 2. PROCESAMIENTO (HPSS)
  armonico, percusivo = dsp.separar_hpss_hifi(y)

  # 3. GUARDADO (Actualizado con nombre dinámico)
  # Ahora cada canción tendrá su propio archivo de salida
  ruta_vocal = os.path.join(carpeta_salida, f"{nombre_base}_vocal_inst.wav")
  ruta_drums = os.path.join(carpeta_salida, f"{nombre_base}_drums.wav")
  
  audio.guardar_wav(ruta_vocal, armonico, sr)
  audio.guardar_wav(ruta_drums, percusivo, sr)

  # 4. VISUALIZACIÓN (Opcional - También con nombre dinámico)
  espectro = dsp.obtener_espectrograma(y)
  ruta_grafico = os.path.join(carpeta_salida, f"analisis_{nombre_base}.png")
  
  plots.graficar_espectrograma(
    espectro, 
    sr, 
    f"Análisis de {nombre_cancion}", 
    ruta_grafico
  )
  
  print(f"✅ Proceso terminado para: {nombre_base}")

if __name__ == "__main__":
  # Ahora podés ir cambiando el nombre acá, correrlo, y se guardarán por separado
  ejecutar_abra_spectral("03 - Gorilla.mp3")