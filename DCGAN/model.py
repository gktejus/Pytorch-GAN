import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

class Generator(nn.Module):
	def __init__(self , d = 128 , channels = 1):
		super(Generator, self).__init__()


		self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0 )
		self.bnom1 = nn.BatchNorm2d(d*8)

		self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1 )
		self.bnom2 = nn.BatchNorm2d(d*4)

		self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1 )
		self.bnom3 = nn.BatchNorm2d(d*2)

		self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1 )
		self.bnom4 = nn.BatchNorm2d(d)


		self.deconv5 = nn.ConvTranspose2d(d,channels,4,2,1 )

	def forward(self, x):
		#print(x.shape)
		x =  x.view(x.shape[0], x.shape[1], 1, 1)
		#print(x.shape)
		x  = F.relu(self.bnom1(self.deconv1(x)))
		#print(x.shape)
		x  = F.relu(self.bnom2(self.deconv2(x)))
		#print(x.shape)
		x  = F.relu(self.bnom3(self.deconv3(x)))
		#print(x.shape)
		x  = F.relu(self.bnom4(self.deconv4(x)))
		#print(x.shape)
		x  = torch.tanh(self.deconv5(x))

		return x

	def weight_init(self, mean , std):
		for m in self._modules:
			normal_init(self._modules[m], mean , std)


class Discriminator(nn.Module):
	def __init__(self , d=128, channels=1):
		super(Discriminator, self).__init__()

		self.conv1 = nn.Conv2d(channels, d,4, 2, 1)

		self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
		self.bnom2 = nn.BatchNorm2d(d*2)

		self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
		self.bnom3 = nn.BatchNorm2d(d*4)

		self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
		self.bnom4 = nn.BatchNorm2d(d*8)

		self.conv5 = nn.Conv2d(d*8, 1,4, 1 , 0)

	def forward(self, x):
		
		x =	F.leaky_relu(self.conv1(x),0.2)
		x = F.leaky_relu(self.bnom2(self.conv2(x)),0.2)
		x = F.leaky_relu(self.bnom3(self.conv3(x)),0.2)
		x = F.leaky_relu(self.bnom4(self.conv4(x)),0.2)

		x = torch.sigmoid(self.conv5(x))


		return x
	def weight_init(self, mean , std):
		for m in self._modules:
			normal_init(self._modules[m], mean , std)	


def normal_init(m , mean , std ):
	# x = 1
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		m.weight.data.normal_(mean , std)
		m.bias.data.zero_()
		# if(x==1):
		# 	print(m)
		# 	print(m.weight.data)
		# 	x+=1
