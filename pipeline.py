import os
# Importamos tus módulos de la carpeta src
from src.core import audio, dsp
from src.utils import plots

def procesar_todo_el_dataset():
  # 1. Definir carpetas
  carpeta_input = os.path.join("data", "input")
  carpeta_output = os.path.join("data", "output")
  os.makedirs(carpeta_output, exist_ok=True)

  # 2. Listar todos los archivos en data/input
  archivos = os.listdir(carpeta_input)
  # Filtramos para que solo tome archivos de audio (evitar carpetas o basura)
  canciones = [f for f in archivos if f.lower().endswith(('.mp3', '.wav', '.flac'))]

  print(f"📂 Encontradas {len(canciones)} canciones para procesar.")

  # 3. El Bucle Mágico (For Loop)
  for i, nombre_cancion in enumerate(canciones):
    print(f"\n🏔️ [{i+1}/{len(canciones)}] Procesando: {nombre_cancion}")
    
    try:
      # Extraemos el nombre sin extensión para el guardado
      nombre_base = os.path.splitext(nombre_cancion)[0]

      # --- Lógica de Procesamiento ---
      ruta_entrada = os.path.join(carpeta_input, nombre_cancion)
      
      # Carga (usando None para procesar todo el tema)
      y, sr = audio.cargar_audio(ruta_entrada, duration=None)

      # Separación HPSS
      armonico, percusivo = dsp.separar_hpss_hifi(y)

      # Guardado dinámico
      ruta_vocal = os.path.join(carpeta_output, f"{nombre_base}_vocal_inst.wav")
      ruta_drums = os.path.join(carpeta_output, f"{nombre_base}_drums.wav")
      
      audio.guardar_wav(ruta_vocal, armonico, sr)
      audio.guardar_wav(ruta_drums, percusivo, sr)

      # Visualización dinámica
      espectro = dsp.obtener_espectrograma(y)
      ruta_grafico = os.path.join(carpeta_output, f"analisis_{nombre_base}.png")
      plots.graficar_espectrograma(espectro, sr, f"Análisis de {nombre_cancion}", ruta_grafico)

      print(f"✅ Completado con éxito: {nombre_base}")

    except Exception as e:
      print(f"❌ Error procesando {nombre_cancion}: {e}")
      continue # Si una falla, sigue con la siguiente

  print("\n🎉 ¡Dataset completo procesado!")

if __name__ == "__main__":
  procesar_todo_el_dataset()