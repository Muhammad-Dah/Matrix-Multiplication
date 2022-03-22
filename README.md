# Matrix-Multiplication

## Testing different ways of multiplying matices in Python
Python's time module provides various time-related functions. We are going to utilize it in order to calculate how long does it take to perform operations. We will cover:
* The CUDA programming model
* Accelerating numerical python code with numba
* Implementing CUDA kernels in python
* Thread synchronization
* shared memory

# Results
![image](https://user-images.githubusercontent.com/37774604/159554830-a6305a54-43d3-4c7a-8562-f0c2b01eb88f.png)

# Summary
* CUDA provides a very powerful framework for easily writing highly scalable multithreaded code.
* Once we have the right mental model about how it works, we can leverage the power of GPUs for performing arbitrary computation.
* Using numba, we can do this directly in Python, and even iterate implementing our GPU code interactively within a jupyter notebook.
* As a bonus, we learned how to accelerate any numerical python function with numba, and squeeze out extra performance gains even without a GPU.

* As we can see, performing the same operation on a GPU gives us a speed-up of 70 times as on CPU. This was still a small computation. For large scale computations, GPUs give us speed-ups of a few orders of magnitude.

* For large scale computations, GPUs give us speed-ups even better than numpy !

# Thanks
