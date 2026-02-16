import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import librosa
from src.core import audio
from src.models import unet
from src.training import engine

# --- CONFIGURACIÓN ---
CARPETA_INPUT = "data/input"   # Donde están los MP3 originales
CARPETA_OUTPUT = "data/output" # Donde están los _drums.wav que generó tu pipeline
EPOCHS = 100                   # Cantidad de vueltas de aprendizaje
DURACION_FRAGMENTO = 10        # Segundos que lee por vez (para no llenar la RAM)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def buscar_pares_de_entrenamiento():
  """
  Esta función hace el trabajo de detective:
  Va a la carpeta de salida y busca archivos que terminen en '_drums.wav'.
  Luego intenta encontrar su canción original correspondiente.
  """
  pares_encontrados = []
  
  # 1. Listamos todo lo que hay en la carpeta de salida
  archivos_salida = os.listdir(CARPETA_OUTPUT)
  
  print(f"📂 Escaneando {CARPETA_OUTPUT}...")
  
  for archivo_drum in archivos_salida:
    # Solo nos interesan los archivos de batería
    if archivo_drum.endswith("_drums.wav"):
      # Obtenemos el nombre base. Ej: "Gorilla_drums.wav" -> "Gorilla"
      nombre_base = archivo_drum.replace("_drums.wav", "")
      
      # 2. Buscamos el original en la carpeta de entrada
      # Esto asume que el archivo original contiene ese nombre base
      for archivo_input in os.listdir(CARPETA_INPUT):
        if nombre_base in archivo_input:
          # ¡Encontramos la pareja! (Cancion Original, Bateria Separada)
          pares_encontrados.append((archivo_input, archivo_drum))
          break
  
  return pares_encontrados

def entrenar_modelo_generalista():
  # 1. Obtenemos la lista de canciones disponibles
  lista_canciones = buscar_pares_de_entrenamiento()
  
  if len(lista_canciones) == 0:
    print("❌ No encontré pares de canciones. Revisa que ejecutaste el pipeline primero.")
    return

  print(f"✅ ¡Entrenamiento listo! Se encontraron {len(lista_canciones)} canciones para estudiar.")
  print(f"   Ejemplos: {lista_canciones[:3]}") # Muestra las primeras 3 para verificar

  # 2. Inicializamos el cerebro
  red = unet.AbraUNet(n_channels=1, n_classes=1).to(DEVICE)
  optimizer = optim.Adam(red.parameters(), lr=0.001)
  criterion = nn.L1Loss()

  print(f"🥊 Iniciando entrenamiento de {EPOCHS} rondas...")

  # 3. Bucle de entrenamiento
  for epoch in range(EPOCHS):
    # A. ELEGIR AL AZAR: En cada vuelta, la IA elige una canción distinta de la lista
    archivo_mix, archivo_target = random.choice(lista_canciones)
    
    try:
      # B. CARGAR FRAGMENTO: Elegimos un punto aleatorio de la canción
      # (Necesitas que tu audio.py acepte 'offset', si no, pon offset=0)
      inicio_random = random.uniform(0, 30) # Primeros 30 seg para asegurar que existe audio
      
      path_mix = os.path.join(CARPETA_INPUT, archivo_mix)
      path_target = os.path.join(CARPETA_OUTPUT, archivo_target)

      # Cargamos solo 10 segundos
      y_mix, _ = audio.cargar_audio(path_mix, offset=inicio_random, duration=DURACION_FRAGMENTO, mono=True)
      y_target, _ = audio.cargar_audio(path_target, offset=inicio_random, duration=DURACION_FRAGMENTO, mono=True)
      
      # C. PROCESAR
      S_mix = np.abs(librosa.stft(y_mix, hop_length=512))
      S_target = np.abs(librosa.stft(y_target, hop_length=512))
      
      tensor_mix = torch.tensor(S_mix).unsqueeze(0).unsqueeze(0).to(DEVICE)
      tensor_target = torch.tensor(S_target).unsqueeze(0).unsqueeze(0).to(DEVICE)

      # D. ENTRENAR
      loss = engine.train_one_epoch(red, optimizer, criterion, tensor_mix, tensor_target)

      # Solo imprimimos cada 5 épocas para no ensuciar la consola
      if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Canción: {nombre_base} - Error: {loss:.5f}")
            
    except Exception as e:
      # Si una canción falla, no detenemos todo, probamos otra en la siguiente vuelta
      #print(f"⚠️ Saltando error en {archivo_mix}: {e}")
      continue

  # 4. GUARDAR
  torch.save(red.state_dict(), "data/output/abra_model_generalist.pth")
  print("💾 ¡Modelo Generalista Guardado!")

if __name__ == "__main__":
  entrenar_modelo_generalista()