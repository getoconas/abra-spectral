import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
  """
  Bloque convencional: (Convolución => BN => ReLU) * 2
  """
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.BatchNorm2d(out_channels),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    return self.conv(x)

class AbraUNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(AbraUNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes

    # --- Bajada (Encoder) ---
    #self.inc = DoubleConv(n_channels, 64)
    #self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
    #self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
    #self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512)) # Cuello de botella
    
    # --- Subida (Decoder) ---
    #self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    #self.conv_up1 = DoubleConv(512, 256)
    
    #self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    #self.conv_up2 = DoubleConv(256, 128)
    
    #self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    #self.conv_up3 = DoubleConv(128, 64)

    # Capa de salida
    #self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    # Reducimos los filtros significativamente para que el Ryzen respire
    self.inc = DoubleConv(n_channels, 16) # Antes 64
    self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(16, 32)) # Antes 128
    self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64)) # Antes 256
    self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128)) # Antes 512 (Bottleneck)
    
    self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv_up1 = DoubleConv(128, 64)
    
    self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.conv_up2 = DoubleConv(64, 32)
    
    self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
    self.conv_up3 = DoubleConv(32, 16)

    self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

  def forward(self, x):
    # Guardar activaciones para las skip-connections
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    
    # Subir y concatenar
    x = self.up1(x4)
    x = self._pad_and_cat(x3, x) # Función auxiliar para ajustar tamaños
    x = self.conv_up1(x)

    x = self.up2(x)
    x = self._pad_and_cat(x2, x)
    x = self.conv_up2(x)

    x = self.up3(x)
    x = self._pad_and_cat(x1, x)
    x = self.conv_up3(x)

    logits = self.outc(x)
    return torch.sigmoid(logits)

  def _pad_and_cat(self, x_target, x_tensor):
    """Ajusta el tamaño de x_tensor para que coincida con x_target y concatena"""
    diffY = x_target.size()[2] - x_tensor.size()[2]
    diffX = x_target.size()[3] - x_tensor.size()[3]
    x_tensor = F.pad(x_tensor, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
    return torch.cat([x_target, x_tensor], dim=1)