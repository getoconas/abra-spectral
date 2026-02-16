# 🏔️ Abra-Spectral

**Abra-Spectral** es un entorno de experimentación en Python para la deconstrucción de señales de audio. El proyecto nace de la curiosidad por separar las "capas" de una canción (stems) mediante procesamiento digital de señales (DSP) y aprendizaje profundo (Deep Learning).

El nombre hace referencia al **Abra**, el paso natural entre las cumbres de Jujuy, simbolizando la apertura de una mezcla musical para revelar sus frecuencias ocultas.

## 🎯 Objetivos del Proyecto
* **Visualización:** Transformar ondas de audio en espectrogramas tratables.
* **Análisis:** Extraer información rítmica y melódica de canciones complejas.
* **Separación:** Implementar una red neuronal (U-Net) para aislar instrumentos (Voz, Batería, Bajo).

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.10+
* **Procesamiento:** Librosa (Análisis de audio), NumPy (Matrices).
* **IA/ML:** PyTorch (Arquitectura de redes neuronales).
* **Visualización:** Matplotlib.

## 🚀 Hoja de Ruta (Roadmap)
1. [ ] Configuración del entorno y carga de archivos.
2. [ ] Transformada de Fourier y visualización de espectrogramas.
3. [ ] Creación de máscaras binarias para separación básica.
4. [ ] Entrenamiento de modelo U-Net con dataset MUSDB18.