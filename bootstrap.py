import numpy as np

# simple implementation of paired bootstrap test
# takes in two functions that score models, the test set, and 
# the number of samples; returns the p value
# here, null hypothesis is that func1 is not significantly 
# better than func2
class PairedBootstrapTest:
	def __init__(self, func1, func2, test_set, num_samples=100000, sample_size=10, threshold=0.01):
		self.func1       = func1
		self.func2       = func2
		self.test_set    = test_set
		self.num_samples = num_samples
		self.sample_size = sample_size
		self.threshold   = threshold
	
	def bootstrap(self):
		delta = self.func1(self.test_set) - self.func2(self.test_set) # how much better is func1 than func2

		s = 0
		for i in range(self.num_samples):
			# draw random bootstrap sample and see how much better func1 is than func2 on this sample
			# by default, it does with replacement 
			bootstrap_sample = np.random.choice(self.test_set, self.sample_size)
			curr_delta = self.func1(bootstrap_sample) - self.func2(bootstrap_sample)
			
			# rule from berg-kirkpatrick et al
			if curr_delta >= 2 * delta:
				s += 1
		
		return s / self.num_samples # p-value
	
	def evaluation(self):
		if self.bootstrap() < self.threshold:
			return 'this result is sufficiently surprising; reject the null hypothesis and conclude that func1 is better than func2'
		else:
			return 'this result is not sufficiently surprising; keep the null hypothesis and conclude that func1 is better than func2'


# dummy example; clearly, if our test set consists of positive integers,
# we expect that foo (which gives the +ve average of the test set) is
# significantly better than bar (which gives the -ve average of the test set)
def foo(x): return sum(x) / len(x)
def bar(x): return -sum(x) / len(x)

bootstrapper = PairedBootstrapTest(foo, bar, np.random.randint(1, 10000, 100))
print(bootstrapper.evaluation())
		
