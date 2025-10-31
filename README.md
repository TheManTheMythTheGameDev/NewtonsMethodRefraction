# Screen-Space Ray Tracing via Newton's Method  - for Accelerated Refractions and Caustics
## Implementation of Newton's Method for rendering refractions and caustics
*Caustics are adapted from https://medium.com/@martinRenou/real-time-rendering-of-water-caustics-59cda1d74aa to use our novel application of Newton's method.*

*Ray marching (for comparison) implemented via https://jcgt.org/published/0003/04/04/ for pixel-perfect results.*

### Dependencies

The only dependency is a small OpenGL wrapper called raylib. Library and header files are included in the repository.

### Building

The Visual Studio solution is ready for building.

### More information

Controls in the app: use the number keys to switch between preset viewpoints (there are currently 2, activated by the '1' and '2' keys).

Snippets of code can be commented/uncommmented to record success and failure rates for either caustics or eye-ray refractions.

Press the 'T' key to begin benchmarking the water shader. Press 'T' again to end benchmarking and print the average computation time. Note that the water shader also contains screen-space reflections and other effects by default, bloating its execution time.

Press the 'K' key to take a screenshot called 'screenshot.png' in the working directory.