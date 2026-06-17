import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def _resolve_device(device=None):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class Square(nn.Module):
    """
    Attivazione: y = x^2
    Nota: evita l'uso in-place se x richiede gradiente su reti complesse,
    per non interferire con l'autograd.
    """
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            # Attenzione: può creare problemi con l'autograd in alcuni casi.
            return x.mul_(x)
        else:
            return x * x

class time_embeddingOLD(torch.nn.Module):
    def __init__(self,F,T,device=None):
      super().__init__()
      device = _resolve_device(device)
      self.T= torch.tensor(T, device=device) # max len of "time"
      self.F= torch.tensor(F, device=device) # fourier flow mode: total lenght = F+1

      self.pi = torch.tensor(torch.pi, device=device)
      self.device = device

    def forward(self,t):
      return self.make_K(t)


    def cos_base(self,mode,t,T):

        return torch.cos(torch.tensor(2.0, device=self.device) *self.pi*t*mode/T).squeeze()

    def sin_base(self, mode,t,T):
        return torch.sin(torch.tensor(2.0, device=self.device)*self.pi*t*mode/T).squeeze()

    def make_K(self, t):
      #F must be odd!!!
        one = torch.tensor([1.0], device=self.device)
        modes = torch.arange(1, (self.F - 1) / 2 + 1, 1, device=self.device)
        return torch.hstack((one, self.cos_base(modes, t, self.T), self.sin_base(modes, t, self.T)))
        #return torch.hstack((torch.tensor([1.0]),self.cos_base(torch.arange(1,(self.F-1)/2+1,1),t,self.T),self.sin_base(torch.arange(1,(self.F-1)/2+1,1),t,self.T)))

class time_embedding(nn.Module):
    def __init__(self, F, T=1, device=None, dtype=torch.float64):
        super().__init__()
        assert F % 2 == 1, "F deve essere dispari (1 + 2*half)"
        self.F = int(F)  # serve come dimensione per la Linear a valle
        device = _resolve_device(device)

        # Registra costanti come buffer (niente grad, si spostano con .to(device))
        self.register_buffer('T', torch.as_tensor(T, dtype=dtype, device=device))
        self.register_buffer('two_pi', torch.tensor(2.0 * math.pi, dtype=dtype, device=device))
        half = (self.F - 1) // 2
        self.register_buffer('modes', torch.arange(1, half + 1, dtype=dtype, device=device))  # [half]
        self.register_buffer('one', torch.ones(1, dtype=dtype, device=device))                # [1]

    def _to_scalar_tensor(self, t):
        """
        Converte t (float/int/tensor con qualunque shape) in un tensore scalare
        sullo stesso device/dtype dei buffer.
        """
        t = torch.as_tensor(t, device=self.T.device, dtype=self.T.dtype)
        if t.numel() != 1:
            t = t.reshape(-1)[0]  # prendi il primo elemento
        return t  # shape: scalar (0-dim)

    def forward(self, t):
        # t scalare (0-dim) sul device/dtype giusto
        t = self._to_scalar_tensor(t)

        # angle: [half]
        angle = (self.two_pi * t * self.modes) / self.T
        cos_part = torch.cos(angle)  # [half]
        sin_part = torch.sin(angle)  # [half]

        # K: [F] = [1] || [half] || [half]
        K = torch.cat((self.one, cos_part, sin_part), dim=0)
        return K  # shape: (F,)

class HyperTimeConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,embedding, rho_init = 1e-5,hidden_dim=16,device=None):
        super().__init__()
        device = _resolve_device(device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.embedding = embedding
        self.F = embedding.F
        # MLP maps scalar → hidden → conv weights ### Qui possiamo aggiungere time embedding tipo fourier o sigmoid come nel caso prima
        # add here embedding as a first layer and fix the nn.Linear input to output dim of embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(self.F, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels * in_channels * kernel_size).to(device) #good initialization weights = 0 and really small bias (like initial guess for rho)
        )
        #self.time_mlp[-1].weight.data.fill_(0.0)
        #self.time_mlp[-1].bias.data.fill_(0.0)

        self.bias = nn.Parameter(torch.ones(out_channels, device=device) * rho_init)
        self.padding = (kernel_size - 1) // 2  # circular 'same' padding

    def forward(self, x, t):
        """
        x: (B, C_in, L)
        t: scalar or (1,) tensor (shared across batch)
        """
 
        # 1. Generate conv kernel weights from time
        weight = self.time_mlp(self.embedding(t))  # (1, C_out * C_in * K)
        weight = weight.view(self.out_channels, self.in_channels, self.kernel_size) #l'output sono i pesi, possiamo anche fare una matrice ( C_out, C_in,  K, hyper) e contrarla con la rete con output hyper

        # 2. Pad input and apply dynamic convolution
        x = F.pad(x, (self.padding, self.padding), mode='circular')
        out = F.conv1d(x, weight, bias=self.bias, padding=0)

        return out



class KISS_HTConv1D(nn.Module):
      def __init__(self, in_channels, out_channels, kernel_size, embedding, rho_init = 1e-5, hidden_dim=16,device=None):
        super().__init__()
        device = _resolve_device(device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

      
        self.embedding = embedding
        self.F = embedding.F

        out_shape =  out_channels * in_channels * kernel_size * self.F

        self.rho = nn.Parameter(torch.randn(out_shape)/out_shape).to(device)
        #self.rho /= (out_shape)
        self.bias = nn.Parameter(torch.zeros(out_channels, device=device))
      
        self.padding = (kernel_size - 1) // 2  # circular 'same' padding

      def get_weights(self,t):
        t_embedd = self.embedding(t)
        weight = torch.mm(self.rho.view(-1,self.F),t_embedd.view(self.F,1)).squeeze()
        return weight.view(self.out_channels, self.in_channels, self.kernel_size)

      def forward(self, x, t):
        """
        x: (B, C_in, L)
        t: scalar or (1,) tensor (shared across batch)
        """
        #if isinstance(t, float) or len(t.shape) == 0:
         #   t = t.view(1)  # ensure shape (1, 1)
        #elif len(t.shape) == 1:
         #   t = t.unsqueeze(0)  # (1, 1)

        # 1. Generate conv kernel weights from time
        weight = self.get_weights(t)/x.shape[-1] 
        
        # 2. Pad input and apply dynamic convolution
        x = F.pad(x, (self.padding, self.padding), mode='circular')
        out = F.conv1d(x, weight, bias=self.bias, padding=0)

        return out

class HyperTimeSeparableConv4D(nn.Module):
    def __init__(self, layer,embedding, in_channels, out_channels, kernel_size,rho_init = 1e-5, hidden_dim=16, permuation_invariance = True,device=None):
        super().__init__()
        device = _resolve_device(device)
        self.kernel_size = kernel_size

        # 4x 1D depthwise convs (no mixing of channels)
        if permuation_invariance:
            layer = layer(in_channels, in_channels, kernel_size,embedding,rho_init, hidden_dim,device)
            self.convs = nn.ModuleList([layer for _ in range(4)]).to(device)
        else:
            self.convs = nn.ModuleList([layer(in_channels, in_channels, kernel_size,embedding,rho_init, hidden_dim,device) for _ in range(4)
        ]).to(device)

        # Final pointwise 1x1 conv to mix channels
        self.channel_mix = nn.Conv3d(in_channels, out_channels, 1).to(device)
        #self.channel_mix.weight.data.fill_(1.0/in_channels/out_channels)
        #self.channel_mix.bias.data.fill_(0.0)

    def forward(self, x,t):
        # x: (batch, channels, D1, D2, D3, D4)
        for axis in range(2, 6):  # Apply 1D convs along D1..D4
            x = self.apply_conv1d_along_axis(x,t, self.convs[axis - 2], axis)

        # Mix channels with a 1x1 conv (interpreting last 3 dims as spatial for 3D conv)
        b, c, d1, d2, d3, d4 = x.shape
        x = x.view(b, c, d1, d2, d3 * d4)
        x = self.channel_mix(x)
        x = x.view(b, -1, d1, d2, d3, d4)
        return x

    def apply_conv1d_along_axis(self, x, t, conv1d, axis):
        # Move the axis to the last position
        x = x.transpose(axis, -1)  # Now x is (..., L)
        orig_shape = x.shape
        batch = x.numel() // x.shape[-1] // x.shape[1]  # b * prod(other_dims)

        x = x.contiguous().view(-1, x.shape[1], x.shape[-1])  # (batch*, channels, L)
        x = conv1d(x,t)
        new_L = x.shape[-1]

        x = x.view(*orig_shape[:-1], new_L)
        x = x.transpose(axis, -1)  # Move axis back to original
        return x


import torch
import torch.nn as nn

class SeparableConv4D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 permutation_invariance=True, device=None):
        super().__init__()
        device = _resolve_device(device)
        self.kernel_size = kernel_size

        if permutation_invariance:
            # un solo conv1d condiviso per tutti gli assi
            shared_layer = nn.Conv1d(
                in_channels, in_channels, kernel_size,
                padding=kernel_size // 2, groups=in_channels
            ).to(device)
            self.convs = nn.ModuleList([shared_layer for _ in range(4)])
        else:
            # layer diversi per ogni asse
            self.convs = nn.ModuleList([
                nn.Conv1d(
                    in_channels, in_channels, kernel_size,
                    padding=kernel_size // 2, groups=in_channels
                ).to(device)
                for _ in range(4)
            ])

        # Pointwise 1x1 conv finale per mescolare i canali
        self.channel_mix = nn.Conv3d(in_channels, out_channels, 1).to(device)
        #self.channel_mix.weight.data.fill_(1.0 / (in_channels * out_channels))
        #self.channel_mix.bias.data.fill_(0.0)

    def forward(self, x):
        # x: (batch, channels, D1, D2, D3, D4)
        for axis in range(2, 6):  # D1..D4
            x = self.apply_conv1d_along_axis(x, self.convs[axis - 2], axis)

        # mix canali con 1x1 conv
        b, c, d1, d2, d3, d4 = x.shape
        x = x.view(b, c, d1, d2, d3 * d4)
        x = self.channel_mix(x)
        x = x.view(b, -1, d1, d2, d3, d4)
        return x

    def apply_conv1d_along_axis(self, x, conv1d, axis):
        # porta l'asse scelto in fondo
        x = x.transpose(axis, -1)  # (..., L)
        orig_shape = x.shape

        # flatten di batch e altre dimensioni
        x = x.contiguous().view(-1, x.shape[1], x.shape[-1])  # (batch*, channels, L)
        x = conv1d(x)
        new_L = x.shape[-1]

        # reshape
        x = x.view(*orig_shape[:-1], new_L)
        x = x.transpose(axis, -1)  # ripristina asse
        return x

def make_hyper_conv_net(layer,embedding, hidden_sizes, kernel_size, in_channels, out_channels,rho_init =1e-5, hidden_mlp=16,device=None):
    '''
    Convolutionaal Neural Netowrk
    hiddens_sizes=[N_filters for hidden layer 1, .... ,N_filters for hidden layer n]
    num hidden layers = len(hidden_sizes)
    '''
    sizes = [in_channels] + hidden_sizes + [out_channels]
    #assert packaging.version.parse(torch.__version__) >= packaging.version.parse('1.5.0')
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    padding_size = (kernel_size // 2)
    net = []
    for i in range(len(sizes) - 1):
        #conv = torch.nn.Conv2d(sizes[i], sizes[i+1], kernel_size, padding=padding_size, stride=1, padding_mode='zeros', bias=use_bias,dilation=1)
        conv = HyperTimeSeparableConv4D(layer, embedding,sizes[i], sizes[i+1], kernel_size=kernel_size,rho_init=rho_init,hidden_dim = hidden_mlp,device=device)
        net.append(conv)

        # se siamo all'ultimo layer, inizializziamo a zero per output iniziale nullo
        if i == len(sizes) - 2:
            nn.init.constant_(conv.channel_mix.weight, 0.0)
            nn.init.constant_(conv.channel_mix.bias, 0.0)
        
        if i != len(sizes) - 2:
            net.append(torch.nn.SiLU())

    net.append(Square()) #attivazione finale
    return torch.nn.Sequential(*net)


def maker_conv_net(hidden_sizes, kernel_size, in_channels, out_channels, device=None):
    device = _resolve_device(device)
    sizes = [in_channels] + hidden_sizes + [out_channels]
    assert kernel_size % 2 == 1, 'kernel size must be odd for PyTorch >= 1.5.0'
    net = []
    for i in range(len(sizes) - 1):
        conv = SeparableConv4D(sizes[i], sizes[i+1], kernel_size=kernel_size, device=device)
        
        # se siamo all'ultimo layer, inizializziamo a zero per output iniziale nullo
        if i == len(sizes) - 2:
            nn.init.constant_(conv.channel_mix.weight, 0.0)
            nn.init.constant_(conv.channel_mix.bias, 0.0)
        
        net.append(conv)
        if i != len(sizes) - 2:
            net.append(nn.SiLU())
    
    net.append(Square())  # attivazione finale
    return nn.Sequential(*net)


class HTCNN(nn.Module):
    def __init__(self,layer,embedding, in_channels, out_channels=1, kernel_size=3, rho_init =1e-5, hidden_sizes=[32, 32, 32], hidden_mlp=16,hyper=True, residual=True,device=None):
        super().__init__()
        device = _resolve_device(device)

        if hyper:
            self.cnn_layers = make_hyper_conv_net(layer,embedding, hidden_sizes, kernel_size, in_channels, out_channels,rho_init, hidden_mlp,device=device)
        else:
            self.cnn_layers = maker_conv_net(hidden_sizes, kernel_size, in_channels, out_channels, device)

    def forward(self, x, t):
        """
        x: (batch, channels, D1, D2, D3, D4)
        t: scalar or (1,) tensor (shared across batch)
        """
        # Pass the time 't' to each HyperTimeSeparableConv4D layer.
        # This requires modifying the forward pass to iterate through layers
        # and pass 't' specifically to the HyperTimeSeparableConv4D layers.
        # A simpler approach for this sequential model is to assume 't' is
        # applied internally within the HyperTimeSeparableConv4D layer.
        # If a more complex architecture (like residual connections) is needed,
        # the forward pass needs to be expanded.

        # For a simple sequential model, we can't directly pass 't' through nn.Sequential.
        # We need a custom forward method.
        for layer in self.cnn_layers:
            if isinstance(layer, HyperTimeSeparableConv4D):
                x = layer(x, t)
            else:
                x = layer(x)
        return x



def test_nets():
    device = 'cpu'
    in_channels = 18
    out_channels = 1
    kernel_size = 3
    L = 16
    test_shape = (10, in_channels, L, L, L, L)
    test_input = torch.randn(test_shape).to(device)
    test_time = torch.tensor([0.7]).to(device) # Example time

    rho_init = 1e-5

    conv = KISS_HTConv1D
    conv1 = HyperTimeConv1D


    embedding = time_embedding(3,1,device)

    print("Model Keep It Simple Stupid Hyper Time Conv 4D")
    model = HTCNN(conv,embedding,in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,hidden_sizes=[32],device=device)
    print(model)

    output = model(test_input, test_time)
    print("Output shape:", output.shape)
    print("Check init:", output.mean()-rho_init)


    print("Model MLP Hyper Time Conv 4D")
    model = HTCNN(conv1,embedding,in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,hidden_sizes=[32],device=device)
    print(model)

    output = model(test_input, test_time)
    print("Output shape:", output.shape)
    print("Check init:", output.mean()-rho_init)
    print('=^.^=')

if __name__ == "__main__":
    test_nets()

    
    device = _resolve_device(device)
