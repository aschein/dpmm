import numpy as np

def main():
	Y = np.zeros(2)
	print Y
	def func(Y):
		print Y + 1
		Y += 1
		print Y
	func(np.zeros(2))
	print Y
if __name__ == "__main__":
	main()