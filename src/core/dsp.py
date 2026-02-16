import librosa
import numpy as np

def separar_hpss_hifi(y):
  print("🔨 Aplicando separación HPSS Hi-Fi...")
  
  # Si es estéreo (2 canales)
  if y.ndim > 1:
    canal_izq = y[0]
    canal_der = y[1]
    
    # Procesar cada canal
    h_L, p_L = librosa.effects.hpss(canal_izq, margin=3.0)
    h_R, p_R = librosa.effects.hpss(canal_der, margin=3.0)
    
    # Unir
    armonico = np.array([h_L, h_R])
    percusivo = np.array([p_L, p_R])
  else:
    # Si es mono
    armonico, percusivo = librosa.effects.hpss(y, margin=3.0)
      
  return armonico, percusivo

def obtener_espectrograma(y):
  # Si es estéreo, convertimos a mono solo para la VISUALIZACIÓN
  if y.ndim > 1:
    y = librosa.to_mono(y)
  
  D = librosa.stft(y)
  return librosa.amplitude_to_db(np.abs(D), ref=np.max)