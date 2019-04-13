# The main purpose of this script is to generate labelled snapshots of Ising configurations
import numpy as np
import os
from matplotlib import pyplot as plt


class IsingChain(object):
	""" In a first simple implementation, we will make snapshots on an 
	L x L square lattice
	"""
	def __init__(self, L):
		self.L = L
		self.conf = 2*np.random.randint(2,size=(L,L))-1
		self.accepted_steps = 0
		self.total_steps = 0
		self.symmetry_breaking = 0.01*(2*np.random.randint(2)-1)

	def get_neighbors(self,i,j):
		return [((i+1) % self.L, (j) % self.L),
		 ((i-1) % self.L, (j) % self.L),
		 ((i) % self.L, (j+1) % self.L),
		 ((i) % self.L, (j-1) % self.L)]



	def take_step(self,beta):
		self.total_steps += 1
		(i,j) = np.random.randint(self.L,size=(2,))
		DeltaE = 2*np.sum([self.conf[i,j] for (i,j) in self.get_neighbors(i,j)])*self.conf[i,j]
		# include a small symmetry breaking field during initialization
		DeltaE -= self.symmetry_breaking*self.conf[i,j]


		if np.exp(-beta*DeltaE) >= np.random.random():
			self.conf[i,j] = -self.conf[i,j]
			self.accepted_steps += 1

	def simulate(self,nsteps,beta):
		for i in range(nsteps):
			self.take_step(beta)

	def generate_imgs(self,beta,n_images):
		# run burn in
		self.simulate(100*self.L**2,beta)
		self.symmetry_breaking = 0
		# generate images
		for i in range(n_images):
			self.simulate(10*self.L**2,beta)
			filename = 'imgs-paramagnet/img{}beta{}.png'.format(int(1E8*np.random.random()),round(beta,3))
			plt.matshow((self.conf+1)/2)
			plt.savefig(filename)
			plt.close()		

				

def main():
	L = 128
	nsteps = 10*L**2
	beta0 = 0.3
	def beta(beta0):
		return beta0*(1+0.5*(np.random.random()-0.5))


	IsingChains = [IsingChain(L) for i in range(200)]
	[x.generate_imgs(beta(beta0),3) for x in IsingChains]

if __name__ == "__main__":
	main()

