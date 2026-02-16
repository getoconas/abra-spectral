import torch
import torch.nn.functional as F

def train_one_epoch(model, optimizer, criterion, tensor_mix, tensor_target):
  model.train()
  optimizer.zero_grad()
  
  # 1. Predicción
  mask_pred = model(tensor_mix)
  
  # 2. Ajuste de tamaño ROBUSTO
  # En lugar de recortar, redimensionamos el target para que coincida con la predicción
  # Esto evita el error de "tensor size 0"
  target_resized = F.interpolate(tensor_target, size=(mask_pred.shape[2], mask_pred.shape[3]), mode='bilinear', align_corners=False)
  mix_resized = F.interpolate(tensor_mix, size=(mask_pred.shape[2], mask_pred.shape[3]), mode='bilinear', align_corners=False)
  
  # 3. Calcular la batería estimada
  bateria_estimada = mix_resized * mask_pred

  # 4. Calcular Loss (Error)
  loss = criterion(bateria_estimada, target_resized)
  
  # 5. Aprender
  loss.backward()
  optimizer.step()
  
  return loss.item()