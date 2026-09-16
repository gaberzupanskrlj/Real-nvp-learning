
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#število točk
n=5000
#naključni koti med 0 do 2pi
theta = torch.rand(n) * 2 * np.pi
#radij območja+šum
r=4+0.3 *torch.randn(n)
#prevtorba iz polarnih kordinat v x,y
x1=r * torch.cos(theta)
x2=r * torch.sin(theta)

x=torch.stack((x1,x2),dim=1)

print(x.shape)
#narišemo cloud(samo za vizualizacijo)
plt.scatter(x[:,0],x[:,1],s=1)
plt.axis('equal')
#plt.show()

class CouplingLayer(nn.Module):

    def __init__(self, flip=False):
        super().__init__()

        self.flip = flip

        # mreža za scale
        self.s_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        # mreža za translate
        self.t_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


    def forward(self, x):

        # izberemo katera koodrinata ostane nespremenjena

        if self.flip:
           x1 = x[:, 1:2]
           x2 = x[:, 0:1]
        else:
           x1 = x[:, 0:1]
           x2 = x[:, 1:2]

        # izračunamo scale in translate
        s = self.s_net(x1)
        t = self.t_net(x1)

        # transformacija x2
        y1 = x1
        y2 = x2 * torch.exp(s) + t

        # kooridnati damo nazaj skupaj v originalni vrstni red
        if self.flip:
             y = torch.cat((y2, y1), dim=1)
        else:
            y = torch.cat((y1, y2), dim=1)

        return y, s, t 
    
    def inverse(self, y):
        # razdelimo vhod na dve polovici
        if self.flip:
            y1 = y[:, 1:2]
            y2 = y[:, 0:1]
        else:
            y1 = y[:, 0:1]
            y2 = y[:, 1:2]

        # izračunamo scale in translate
        s = self.s_net(y1)
        t = self.t_net(y1)

        # inverzna transformacija y2
        x1 = y1
        x2 = (y2 - t) * torch.exp(-s)

        # združimo nazaj
        if self.flip:
            x = torch.cat((x2, x1), dim=1)
        else:
            x = torch.cat((x1, x2), dim=1)

        return x


    
    
class RealNVP(nn.Module):
      def __init__(self):
            super().__init__()
#imamo 6 Coupling layerjev, ki se izmenjujejo flip in non-flip
            self.layers = nn.ModuleList([
                CouplingLayer(flip=False),
                CouplingLayer(flip=True),
                CouplingLayer(flip=False),
                CouplingLayer(flip=True),
                CouplingLayer(flip=False),
                CouplingLayer(flip=True),
                CouplingLayer(flip=False),
                CouplingLayer(flip=True),
                CouplingLayer(flip=False), 
                CouplingLayer(flip=True)
                ])
      def forward(self, x):
          z = x

          log_det = x.new_zeros(x.shape[0])
          for layer in self.layers:
              z, s, t= layer(z)
              log_det=log_det + s.squeeze(1)

          return z, log_det
      def inverse(self, z):
            x = z
            for layer in reversed(self.layers):
                x = layer.inverse(x)
            return x
     

import math
def real_nvp_loss(x, model):
    z, log_det = model(x)
    #računamo log verjetnost za standardno normalno porazdelitev
    log_prob_z = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * math.log(2 * math.pi) * z.shape[1]
    #log verjetnost za x
    log_prob_x = log_prob_z + log_det
    #negativna log verjetnost
    loss = -torch.mean(log_prob_x)
    return loss
model = RealNVP()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    #računamo izgubo
    loss = real_nvp_loss(x, model)
    loss.backward()
    #posodimo parametre
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

with torch.no_grad():

    z, log_det = model(x)
    x_back = model.inverse(z)

    print("x shape:", x.shape)
    print("z shape:", z.shape)
    print("x_back shape:", x_back.shape)

    plt.figure(figsize=(12, 4))

    # original
    plt.subplot(1, 3, 1)
    plt.scatter(x[:, 0], x[:, 1], s=1)
    plt.title("Original Data")
    plt.axis("equal")

    # latentni prostor
    plt.subplot(1, 3, 2)
    plt.scatter(z[:, 0], z[:, 1], s=1)
    plt.title("Latent Space (z)")
    plt.axis("equal")

    # rekonstruirani podatki
    plt.subplot(1, 3, 3)
    plt.scatter(x_back[:, 0], x_back[:, 1], s=1)
    plt.title("Reconstructed Data (x_back)")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()

print("mean:")
print(z.mean(dim=0))

print("std:")
print(z.std(dim=0))
with torch.no_grad():
    z, _ = model(x)


with torch.no_grad():
    z, _ = model(x)
    x_back = model.inverse(z)

    print("Mean z:", z.mean(dim=0))
    print("Std z:", z.std(dim=0))

    print("Covariance:")
    print(torch.cov(z.T))

    print(
        "Max reconstruction error:",
        torch.max(torch.abs(x - x_back)).item()
    )

# 3. NOV EKSPERIMENT:
# naredimo popolnoma nov Gaussian
with torch.no_grad():
    z_new = torch.randn(5000, 2)

    # Gaussian -> naučena podatkovna porazdelitev
    x_new = model.inverse(z_new)

# 4. narišemo rezultat
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(z_new[:, 0], z_new[:, 1], s=1)
plt.title("New Gaussian samples")
plt.axis("equal")

plt.subplot(1, 2, 2)
plt.scatter(x_new[:, 0], x_new[:, 1], s=1)
plt.title("Generated samples")
plt.axis("equal")
plt.show()
with torch.no_grad():

    current = x
    states = [current.clone()]

    for layer in model.layers:
        current, _, _ = layer(current)
        states.append(current.clone())

fig, axes = plt.subplots(2, 4, figsize=(14, 7))

with torch.no_grad():

    current = x
    states = [current.clone()]

    for layer in model.layers:
        current, _, _ = layer(current)
        states.append(current.clone())

n_plots = len(states)

cols = 4
rows = math.ceil(n_plots / cols)

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(14, 3.5 * rows)
)

axes = axes.flatten()

for i, ax in enumerate(axes):

    if i >= len(states):
        ax.axis("off")
        continue

    ax.scatter(
        states[i][:, 0],
        states[i][:, 1],
        s=1
    )

    if i == 0:
        ax.set_title("Original x")
    else:
        ax.set_title(f"After layer {i}")

    ax.axis("equal")

plt.tight_layout()
plt.show()

#generate 100 random two dimensional points