import numpy as np

class BatchNormalization():
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape =None

		self.running_mean = running_mean
		self.running_var = running_var

		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None

	def forward(self, input_data, is_train=True):
		self.input_shape = input_data.shape
		if input_data.ndim != 2:
			N, C, H, W = input_data.shape
			input_data = input_data.reshape(N, -1)

		out = self.__forward(input_data, is_train)
	
		return out.reshape(*self.input_shape)

	def __forward(self, input_data, is_train):
		if self.running_mean is None:
			N, D = input_data.shape
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)

		if is_train:
			mu = input_data.mean(axis=0)
			xc = input_data - mu
			var = np.mean(xc**2, axis=0)
			std = np.sqrt(var + 10e-7)
			xn= xc / std
   
			self.batch_size = input_data.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
		else:
			xc = input_data - self.running_mean
			xn = xc / ((np.sqrt(self.running_var + 10e-7)))
		
		out = self.gamma * xn + self.beta
		return out

	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
		
		dx = self.__backward(dout)
  
		dx = dx.reshape(*self.input_shape)
		return dx
	
	def __backward(self, dout):
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(self.xc* dout, axis=0)
		dxn = self.gamma * dout
		dxc = dxn / self.std
		dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
		dvar = 0.5 * dstd / self.std
		dxc += (2.0 / self.batch_size) * self.xc * dvar
		dmu = np.sum(dxc, axis=0)
		dx = dxc - dmu / self.batch_size

		self.dgamma = dgamma
		self.dbeta = dbeta
  
		return dx
    