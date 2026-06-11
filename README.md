<h1>Simulating Quantum Field Fluctuations in Minkowski Space</h1>

Python simulations of quantum field fluctuations for a Klein-Gordon field in Minkowski space.

This repository contains code developed for a PHYS 785 project at the University of Waterloo. The goal of the project was to numerically sample and visualize fluctuations of a scalar quantum field by decomposing the field into Fourier sine modes, treating each mode as a decoupled harmonic oscillator, and reconstructing field configurations from sampled mode amplitudes.

A detailed write-up of the project is included in `Simulating Quantum Field Fluctuations Project FINAL.pdf`.

<br>

<h2>🔹 Project Overview</h2>

This project studies quantum fluctuations of the Klein-Gordon field in flat spacetime.

To make the problem numerically tractable, the field is placed in a finite box with Dirichlet boundary conditions. A Fourier sine expansion is then used to decompose the field into discrete modes. Each mode behaves like an independent harmonic oscillator with a frequency determined by its mode number and the mass of the field.

The code samples amplitudes for these modes from their probability distributions and then reconstructs approximate field configurations. These configurations are visualized using contour plots of field slices such as phi(x, y, z = constant).

<br>

<h2>🔹 Main Features</h2>

* Simulates quantum fluctuations of a Klein-Gordon scalar field
* Uses Fourier sine modes in a finite box with Dirichlet boundary conditions
* Treats field modes as decoupled harmonic oscillators
* Samples mode amplitudes from ground-state probability distributions
* Reconstructs approximate field configurations from finitely many modes
* Produces contour plots of field fluctuations in two-dimensional slices
* Explores how changing mass, number of modes, and dimension affects the field fluctuations
* Includes preliminary exploration of excited modes and wave-packet-like states

<br>

<h2>🔹 Physics and Numerical Methods</h2>

The Klein-Gordon field is expanded in discrete Fourier sine modes after imposing infrared regularization in a finite box. This turns the field into a collection of independent harmonic oscillator degrees of freedom.

For the ground state, each mode has a Gaussian probability distribution for its amplitude. The code samples these amplitudes and combines the modes to approximate the field configuration.

The project then studies how the resulting fluctuations change as physical and numerical parameters are varied. In particular, the simulations examine the effects of the field mass, the number of modes included in the reconstruction, and the difference between a genuinely two-dimensional field and a two-dimensional slice of a three-dimensional field.

<br>

<h2>🔹 Repository Contents</h2>

This repository includes code for:

* sampling Fourier-mode amplitudes
* reconstructing scalar field configurations
* plotting two-dimensional field slices
* comparing different field masses
* comparing different numbers of included modes
* comparing two-dimensional fields with two-dimensional slices of three-dimensional fields
* exploring simple excited-state cases

The accompanying report, `Simulating Quantum Field Fluctuations Project FINAL.pdf`, explains the derivation, implementation, results, limitations, and suggested directions for future work.

<br>

<h2>🔹 Results</h2>

The simulations show that increasing the mass of the field decreases the amplitude of the fluctuations. This agrees with the harmonic-oscillator picture, since increasing the mass increases the mode frequencies and narrows the probability distributions for the mode amplitudes.

The simulations also show that increasing the number of included Fourier modes produces more detailed and higher-amplitude fluctuation patterns. With only a small number of modes, the contour plots are relatively simple and structured. With many modes, the plots become more irregular and resemble the expected behaviour of vacuum field fluctuations.

The project also compares a true two-dimensional field with a two-dimensional slice of a three-dimensional field. These are not equivalent, since the Fourier expansion, normalization factors, and mode frequencies differ between the two cases.

A preliminary excited-state case was also considered. Exciting some of the harmonic oscillator modes changes the probability distributions and can increase the likelihood of larger field amplitudes.

<br>

<h2>🔹 Current Limitations</h2>

This repository should be treated as research and course-project code rather than a finished software package.

The simulations approximate the field using only a finite number of modes, so the results depend on the chosen mode cutoff. The excited-state and wave-packet cases are also preliminary and were explored only for a limited number of excitations.

Future improvements could include:

* improving the sampling method for excited-state distributions
* studying higher excitation numbers
* testing convergence as the number of modes increases
* comparing different mode-selection schemes
* extending the simulation to curved spacetimes such as FRW spacetime
* refactoring the code into a cleaner package structure

<br>

<h2>🔹 Dependencies</h2>

The code was developed in Python using standard scientific-computing libraries, including:

* NumPy
* SciPy
* Matplotlib

Additional dependencies may be required depending on which script is being run.

<br>

<h2>🔹 How to Run</h2>

Clone the repository:

`git clone https://github.com/CSampaio101/QFT-Project.git`

Enter the repository:

`cd QFT-Project`

Install the main dependencies:

`pip install numpy scipy matplotlib`

Run the desired Python script from the repository.

The exact script may depend on which calculation or plot you want to generate.

<br>

<h2>🔹 Author</h2>

Cristiano Sampaio
MSc Physics, University of Waterloo
