# CUDA
The Golden Ratio
The ratio of successive elements of the sequence, i.e. F(n+1) / F(n). It turns out that this ratio tends towards a fixed value, as the Fibonacci numbers get larger. Moreover, this particular value is very well-known to mathematicians through the ages. It is known as the golden ratio, and is given by

gratio=[fiblist[i] / float(fiblist[i-1]) for i in range(2,len(fiblist))]
print gratio

Therefore, this ratio can be utilised to generate a fibonacci sequence without actually depending on the previous numbers in the sequence.

		double phi = 1.618033988749895;  // Golden Ratio
	fiblist[i] = round((pow(phi, i) - pow(-phi, -i)) / sqrt((double)5));
