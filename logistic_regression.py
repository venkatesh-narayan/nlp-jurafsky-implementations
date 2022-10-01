import numpy as np
import sys

class LogisticRegression:
	def __init__(self, X_train, y_train, X_val, y_val, 
				 hidden_dim, num_epochs=10, lr=1e-3, batch_size=32):		
		# assuming all X's and y's are 2d
		self.weights = np.random.randn((X_train.shape[1], hidden_dim))
		self.bias = np.random.randn((hidden_dim, ))

		self.lr = lr
		self.num_epochs = num_epochs

		def batch(X, y, bs):
			batches = []
			for i in range(0, len(y) // bs + 1, bs):
				start = i
				end = min(i + bs, len(y))

				batches.append((X[start:end], y[start:end]))

			return batches

		self.train_batches = batch(X_train, y_train, batch_size)
		self.val_batches = batch(X_val, y_val, batch_size)
	
	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def forward(self, x):
		# [bs, input_dim] x [input_dim, hidden_dim] = [bs, hidden_dim]
		intermediate = x @ self.weights + self.bias
		return self.sigmoid(intermediate)
	
	# cross entropy loss
	def loss(self, y_pred, y):
		return -(np.dot(y, np.log(y_pred)) + np.dot(1 - y, np.log(1 - y_pred)))
	
	def grad_w(self, x, y, y_pred):
		return (y_pred - y) * x
	
	def grad_b(self, y, y_pred):
		return y_pred - y
	
	# mini batch gradient descent
	def sgd(self):
		for x_batch, y_batch in self.train_batches:
			y_pred = self.forward(x_batch)
			self.weights -= self.lr * self.grad_w(x_batch, y_batch, y_pred)
			self.bias -= self.lr * self.grad_b(y_batch, y_pred)

	def train(self, num_epochs):
		for epoch in range(num_epochs):
			self.sgd()
			eval_accuracy, eval_loss = self.eval()
			print(f'epoch {epoch}: val accuracy: {round(eval_accuracy, 3)}, val_loss: {round(eval_loss, 3)}')


	def eval(self):
		eval_loss = 0.0
		num_correct = 0
		total = 0

		for x_batch, y_batch in self.val_batches:
			y_pred = self.forward(x_batch)
			eval_loss += self.loss(y_pred, y_batch)
			num_correct = sum(y_pred == y_batch)
			total += len(y_batch)

		return num_correct / total, eval_loss


if __name__ == '__main__':
	# find sample train and val
	X_train, y_train, X_val, y_val = sys.argv[1:]
	lr = LogisticRegression(X_train, y_train, X_val, y_val, 1)
	lr.train()




