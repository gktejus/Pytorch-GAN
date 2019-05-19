import torch
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms , datasets
from utils import real_loss , fake_loss , load_data , save_weights , save_samples
from model import Generator , Discriminator
import numpy as np 



batch_size = 16
img_size = 64
num_workers = 0 


device = 'cuda'
DATASET = 'MNIST'

train_loader = load_data(DATASET, img_size, batch_size, num_workers)

# data  = iter(train_loader)
# img , label = data.next()
# plt.imshow(img[0].squeeze())
# plt.show()

G = Generator(128, 1)
D = Discriminator(128 , 1)

G.weight_init(0,0.02)
D.weight_init(0,0.02)


G =nn.DataParallel(G)		
D =nn.DataParallel(D)

G = G.cuda()
D = D.cuda()

d_opt =optim.Adam(D.parameters(),lr = 0.002, betas = (0.5, 0.99))
g_opt =optim.Adam(G.parameters(),lr = 0.002, betas = (0.5, 0.99))

num_epochs  = 20 
print_every = 400
save_every  = 800

for epoch in range(num_epochs):
	print("Epoch {}".format(epoch))

	D_loss = []
	G_loss = []

	for batch_i,(real_img , _) in enumerate(train_loader):
		real_img = real_img.to(device)
		sample_size = real_img.size(0)
		
		d_opt.zero_grad()
		#print(real_img.shape)
		d_real = D(real_img)
		
		d_real_loss = real_loss(d_real,smooth=True, cuda=True)
		fake_z = torch.from_numpy(np.random.randn(sample_size, 100)).type(dtype=torch.FloatTensor).to(device)
		#fake_z = fake_z.view(-1, 100, 1 , 1)
		
		# print(fake_z.shape)
		fake_img = G(fake_z)
		#print(fake_img.shape)
		d_fake = D(fake_img)
		d_fake_loss = fake_loss(d_fake, cuda=True)

		d_loss = d_fake_loss +d_real_loss

		d_loss.backward()
		d_opt.step()
		D_loss.append(d_loss.item())


		g_opt.zero_grad()

		fake_z = torch.from_numpy(np.random.randn(sample_size, 100)).type(dtype=torch.FloatTensor).to(device)
		#fake_z = fake_z.view(-1, 100, 1 , 1)
		fake_img  = G(fake_z)
		d_fake = D(fake_img)
		g_loss = real_loss(d_fake, smooth=True, cuda=True)
		g_loss.backward()
		g_opt.step()
		G_loss.append(g_loss.item())
		if batch_i % print_every == 0:
			print('D loss: {:.5f}\tG loss: {:.5f}'.format(np.mean(D_loss), np.mean(G_loss)))
			
		if (batch_i%100==0):
			save_weights(epoch, batch_i, D_loss, G_loss, G)
			save_samples(G, epoch, batch_i, num_samples=25, DATASET = 'MNIST', root_path = './Results')
		

			