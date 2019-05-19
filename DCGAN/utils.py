import torch.nn as nn
import torch 
from torchvision import datasets , transforms
import os
import numpy as np
import matplotlib.pyplot as plt


device = 'cuda'


def real_loss(out,smooth = True, cuda = False):
	batch_size = out.size(0)

	if(smooth):
		lab = torch.ones(batch_size)*0.9
	else:
		lab = torch.ones(batch_size)
	if(cuda):
		lab = lab.to(device)
	criterion = nn.BCELoss()
	loss = criterion(out.squeeze(), lab)

	return loss


def fake_loss(out, cuda = False):
	batch_size = out.size(0)

	lab  = torch.zeros(batch_size)
	if(cuda):
		lab = lab.to(device)

	criterion = nn.BCELoss()
	loss = criterion(out.squeeze(), lab)

	return loss


def load_data (DATASET, img_size , batch_size , num_workers):
	if (DATASET=="MNIST"):
		tfom = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
		train_data = datasets.MNIST('./', transform = tfom , train = True , download = True)
		train_loader = torch.utils.data.DataLoader(train_data , batch_size = batch_size, num_workers = 0 )
	else:
		tfom = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
		train_data = datasets.ImageFolder('./', transform = True)
		train_loader = torch.utils.data.DataLoader(train_data , batch_size = batch_size, num_workers = 0 )
	return train_loader	

def save_weights(epoch , batch_i, D_loss , G_loss , G , root_path  ='./Results'):
	if(os.path.isdir(root_path)==False):
		os.mkdir(root_path)
	epoch_path = os.path.join(root_path, "epoch"+str(epoch))
	if (os.path.isdir(epoch_path)==False):
		os.mkdir(epoch_path)
	batch_path = os.path.join(epoch_path, "batch_"+str(batch_i))
	if (os.path.isdir(batch_path)==False):
		os.mkdir(batch_path)
	f_name ='D_{:.2f}G_{:.2f}.pt'.format(np.mean(D_loss), np.mean(G_loss))
	save_path= os.path.join(batch_path, f_name)	
	torch.save(G, save_path)	

def save_samples(G ,epoch , batch_i ,  num_samples = 16, DATASET = 'MNIST', root_path = './Results'):
	z = torch.from_numpy(np.random.randn(num_samples, 100)).type(dtype=torch.FloatTensor).cuda()
	G.eval()
	fake_imgs = G(z).detach().cpu().numpy()
	fig = plt.figure()
	if(os.path.isdir(root_path)==False):
		os.mkdir(root_path)
	epoch_path = os.path.join(root_path, "epoch"+str(epoch))
	if (os.path.isdir(epoch_path)==False):
		os.mkdir(epoch_path)
	batch_path = os.path.join(epoch_path, "batch_"+str(batch_i))
	if (os.path.isdir(batch_path)==False):
		os.mkdir(batch_path)
	for z in range(num_samples):
		# sample_img = fake_img_num[z]
		# b = sample_img.transpose()
		#b = b.astype(np.uint8)
		b = fake_imgs[z]
		dim = int(np.sqrt(num_samples))
		ax = fig.add_subplot(dim, dim, z+1, xticks=[], yticks=[])
		if(DATASET=='MNIST'):
			plt.imshow(b.squeeze(), cmap= 'gray')
		else:
			plt.imshow(np.transpose(b ,(1,2,0)))	
		img_save_path = os.path.join(batch_path, str(z)+'.jpg')
	plt.savefig(img_save_path, dpi=200  )
	plt.close(fig)
	#print(b.shape)
		# plt.imshow(b.squeeze())
		# plt.show()
	G.train()

