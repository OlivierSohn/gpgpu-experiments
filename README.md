# What is it?

This is a project to experiment [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) using [OpenCL](https://fr.wikipedia.org/wiki/OpenCL).

Editing the [main.cpp](main.cpp) file will let you switch between the different examples.

The code of the first example is a slightly modified version of [this excellent tutorial](https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/).

# Why?

First, parallelizing algorithms is fun!

And in another project, I implemented a convolution reverb algorithm using ffts. Ffts are the bottleneck of the implementation, so I wanted to see if using the GPU to do the ffts would improve the performance, and by how much.

# How?

As always, I try to tackle "one small problem at a time":

First, I started just with the  [tutorial](https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/), and verified I could make it run on my machine.

Then, I iteratively complexified the tutorial, adding one little new aspect at a time, before doing the fft (cooley-tukey algorithm without bitreversal of the input).

At every iteration, I was checking that the kernel code was behaving as intended by comparing the results with an equivalent cpu-based implementation.

And using the environment variable `CL_LOG_ERRORS=stdout` made debugging kernel compilation errors a lot easier!

# Next Steps

* use images to have faster access to global memory:
  * To have faster read only access to inputs, use an image + float4 read_imagef
  * To have fater write to output, use an image + write_imagef
  * read/write images are opencl 2.0 only, but in practice passing the image twice with different
qualifiers can work, depending on the driver + hardware.
* To do big ffts using local memory, compute local levels in chunks, and when writing back, interleave the data
and redo the same thing (except that twiddle factors indinces computation needs to adapt) until all levels are done.
Finallly, reorder global memory (or let the cpu do it?)
* use inter-group synchronization to be able to use multiple work groups for a single fft.
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.172.2399&rep=rep1&type=pdf 
https://www.doc.ic.ac.uk/~afd/homepages/papers/pdfs/2016/OOPSLA.pdf
* Compare with other (open source) fft implementations on the gpu (for example, https://github.com/clMathLibraries/clFFT)
* Implement in-place fft.

# Platforms

Using [CMake](https://cmake.org/) you can build and run it on a recent OSX.

Other platforms are not supported, but I think it's just a matter of making the `CMakeLists.txt` more general regarding the way to link to the [OpenCL](https://fr.wikipedia.org/wiki/OpenCL) library.

# Contributions

PRs are welcome, for example to generalize the `CMakeLists.txt` file to make it build and run on Linux or Windows.
