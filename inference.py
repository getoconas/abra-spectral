import torch
import numpy as np
import librosa
import soundfile as sf
import os
import torch.nn.functional as F

# Importamos tus módulos
from src.core import audio
from src.models import unet

# --- CONFIGURACIÓN ---
# Ruta del cerebro que acabas de entrenar
MODELO_ENTRENADO = os.path.join("data", "output", "abra_model_generalist.pth")

# ¡IMPORTANTE! Cambia esto por una canción NUEVA (que la IA no haya estudiado)
# Por ejemplo, la de Foster the People que usaste al principio
CANCION_NUEVA = "05 - Forrest Gump.mp3" 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def separar_con_ia():
  print(f"🚀 Iniciando inferencia en {DEVICE} con modelo Lite...")

  # 1. Cargar el modelo (La arquitectura debe ser IGUAL a la del entrenamiento)
  # Como usamos la versión Lite en unet.py, al instanciarla aquí ya viene con los 16 canales
  print("🧠 Cargando cerebro...")
  try:
    red = unet.AbraUNet(n_channels=1, n_classes=1).to(DEVICE)
    red.load_state_dict(torch.load(MODELO_ENTRENADO, map_location=DEVICE))
    red.eval() # Modo evaluación (apaga el aprendizaje y congela las neuronas)
  except Exception as e:
    print(f"❌ Error cargando el modelo: {e}")
    return

  # 2. Cargar la canción nueva
  ruta_input = os.path.join("data", "input", CANCION_NUEVA)
  if not os.path.exists(ruta_input):
    print(f"❌ No encuentro la canción: {ruta_input}")
    return

  print(f"🎧 Cargando audio: {CANCION_NUEVA}...")
  # Cargamos 30 segundos para probar rápido (puedes poner None para toda la canción)
  y, sr = audio.cargar_audio(ruta_input, duration=30, mono=True)
  
  # 3. Preprocesamiento (STFT)
  # Usamos hop_length=512 igual que en el entrenamiento final
  D = librosa.stft(y, hop_length=512)
  S_mag, S_phase = librosa.magphase(D) # Separamos Magnitud (Imagen) y Fase (Sonido)

  # Convertir a Tensor
  tensor_input = torch.tensor(S_mag).unsqueeze(0).unsqueeze(0).to(DEVICE)

  # 4. La IA hace su magia
  print("🔮 Separando batería...")
  with torch.no_grad():
    mask_pred = red(tensor_input)
  
  # 5. Reconstrucción
  # A veces la IA devuelve un tamaño ligeramente distinto, ajustamos con interpolación
  mask_final = F.interpolate(
    mask_pred, 
    size=(S_mag.shape[0], S_mag.shape[1]), 
    mode='bilinear', 
    align_corners=False
  ).squeeze().cpu().numpy()

  # --- 🛠️ NUEVO: PUERTA DE RUIDO Y LIMPIEZA ---
  # Cualquier valor de la máscara menor al umbral se silencia (vuelve a 0)
  # Probá con 0.1 o 0.15 para empezar. Si es muy alto, perdés platillos.
  umbral = 0.12 
  mask_final[mask_final < umbral] = 0.0

  # Aplicamos un "Contraste" (opcional): los valores altos se refuerzan
  # Esto ayuda a que el golpe de batería sea más seco y definido
  mask_final = np.power(mask_final, 1.2)
  # --------------------------------------------

  # Aplicamos la máscara a la magnitud original
  bateria_espectro = S_mag * mask_final
  
  # Reconstruimos el audio usando la fase original (ISTFT)
  y_bateria = librosa.istft(bateria_espectro * S_phase, hop_length=512)

  # Cortamos frecuencias por debajo de 30Hz que suelen ser puro ruido subsónico
  #y_bateria = librosa.effects.preemphasis(y_bateria)

  # 6. Guardar
  ruta_salida = os.path.join("data", "output", "resultado_ia_drums.wav")
  sf.write(ruta_salida, y_bateria, sr)
  print(f"✅ ¡Éxito! Escucha el resultado en: {ruta_salida}")

if __name__ == "__main__":
  separar_con_ia()