# What is it?

This is a project to experiment with [GPGPU](https://en.wikipedia.org/wiki/General-purpose_computing_on_graphics_processing_units) using [OpenCL](https://fr.wikipedia.org/wiki/OpenCL).

Editing the `main.cpp` file will let you switch between the different examples.

The code of the first example comes from [this excellent tutorial](https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/).

# Why?

First, parallelizing algorithms is fun!

And in another project, I implemented a convolution reverb algorithm using ffts. Ffts are the bottleneck of the implementation, so I wanted to see if using the GPU to do the ffts would improve the performance, and by how much.

# How?

As always, I try to tackle "one small problem at a time":

First, I started just with the  [tutorial](https://www.eriksmistad.no/getting-started-with-opencl-and-gpu-computing/), and verified I could make it run on my machine.

Then, I iteratively complexified the tutorial, adding one little new aspect at a time, before doing the fft (which is still not yet complete: bit reversal of the input is not handled).

At every iteration, I was checking that the kernel code was behaving as intended by comparing the results with an equivalent cpu-based implementation.

And using the environment variable `CL_LOG_ERRORS=stdout` made debugging kernel compilation errors a lot easier!

# Next Steps

- Find a way to do ffts of a size that is bigger than the number of threads on the GPU.
- See if computing fft twiddle factors on the fly is any faster than fetching them from memory. (It would also make the algorithm use less memory)
- Use memory that is closer to the GPU threads to improve performance.
- See if there are other (open source) fft implementations on the gpu, and compare approaches.

# Platforms

Using [CMake](https://cmake.org/) you can build and run it on a recent OSX.

Other platforms are not supported, but I think it's just a matter of making the `CMakeLists.txt` more general regarding the way to link to the [OpenCL](https://fr.wikipedia.org/wiki/OpenCL) library.

## Paths to kernel files

The paths to the kernel (.cl) files are hardcoded : you will need to edit them to match the location of your files on your harddrive.

# Contributions

PRs are welcome, for example to generalize the `CMakeLists.txt` file to make it build and run on Linux or Windows.
