import torch.nn as nn
import torch 
from torch.utils.data import DataLoader

class Network(nn.Module):
	"""This class creates a network given the right parameters

	:param n_features: 
		number of features in input data
	:param channels:
		a list of channels for all layers in the network
	:param act_fn:
		activation function to be used for all linear layers
	:param batch_norm:
		bool value, if True use Batch Norm after each linear layer
	:param dropout:
		bool value, if True use dropout but no dropout in bottleneck
	"""
	def __init__(self, n_features, channels, act_fn = nn.ReLU(), batch_norm = False, dropout = False):
		super(Network, self).__init__()

		self.channels = channels
		self.encoder = nn.ModuleList()
		self.decoder = nn.ModuleList()
		self.channels.insert(0, n_features)
		decoder_chn = channels[::-1]

		for i in range(len(self.channels) - 1):
			self.encoder.append(nn.Linear(self.channels[i], self.channels[i+1]))
			if batch_norm:
				self.encoder.append(nn.BatchNorm1d(self.channels[i+1]))
			self.encoder.append(act_fn)
			if dropout and i < (len(channels) - 2):
				# dont use dropout in last layer of encoder
				self.encoder.append(nn.Dropout(0.20))

			self.decoder.append(nn.Linear(decoder_chn[i], decoder_chn[i+1]))
			if batch_norm:
				self.decoder.append(nn.BatchNorm1d(decoder_chn[i+1]))
			self.decoder.append(act_fn)
			if dropout and i > 0:
				# dont add dropout in first layer of decoder
				self.decoder.append(nn.Dropout(0.20))

		# remove last layer's activation fn, dropout and Batch norm from decoder
		if dropout:
			self.decoder = self.decoder[:-3]
		else:
			self.decoder = self.decoder[:-2]

	def __repr__(self):
		return 'Layers channels {}'.format(self.channels)

	def forward(self, x):
		"""gets the input tensor and returns output after one forward pass through network"""
		for layer in self.encoder:
			x = layer(x)

		for layer in self.decoder:
			x = layer(x)

		return x
		

class AE_3D_200(nn.Module):
	"""This class creates a particular network, it has been taken from Skelpdar's repository.
	It is required to fine tune the pre-trained model using this network. """
	def __init__(self, n_features=4):
	    super(AE_3D_200, self).__init__()
	    self.en1 = nn.Linear(n_features, 200)
	    self.en2 = nn.Linear(200, 100)
	    self.en3 = nn.Linear(100, 50)
	    self.en4 = nn.Linear(50, 3)
	    self.de1 = nn.Linear(3, 50)
	    self.de2 = nn.Linear(50, 100)
	    self.de3 = nn.Linear(100, 200)
	    self.de4 = nn.Linear(200, n_features)
	    self.tanh = nn.Tanh()

	def encode(self, x):
	    return self.en4(self.tanh(self.en3(self.tanh(self.en2(self.tanh(self.en1(x)))))))

	def decode(self, x):
	    return self.de4(self.tanh(self.de3(self.tanh(self.de2(self.tanh(self.de1(self.tanh(x))))))))

	def forward(self, x):
	    z = self.encode(x)
	    return self.decode(z)

	def describe(self):
	    return 'in-200-100-50-3-50-100-200-out'

# this function returns pytorch dataloaders
def get_data(train_ds, valid_ds, bs, test_ds = None):
	test_dl = None
	if test_ds is not None:
	        test_dl = DataLoader(test_ds, batch_size = len(test_ds), pin_memory = True)
	return (
	    DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True),
	    DataLoader(valid_ds, batch_size=bs , pin_memory=True),
	    test_dl
	)

def std_error(x, axis=None, ddof=0):
    return np.nanstd(x, axis=axis, ddof=ddof) / np.sqrt(2 * len(x))



