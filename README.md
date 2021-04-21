# Perceptual evolution: How the spatially explicit interplay of biological and environmental factors shapes resource uptake #
**Team members:** Anshuman Swain<sup>1,* </sup>, Tyler Hoffman<sup>1,* </sup>, Kirtus Leyba<sup>2 </sup>, and William F Fagan<sup>1</sup>

<sup>1</sup>Department of Biology, University of Maryland, College Park, MD 20742, USA;
<sup>2</sup>Biodesign Institute, Arizona State University, Tempe, AZ 85281, USA;
<sup>*</sup>contributed equally


**Keywords:** Perception, Agent-based model, Evolution



## Brief introduction to the project ##
Perception is central to an individual’s survival as it affects its ability to gather resources. Consequently, the costs associated with the process of perception are partially shaped by resource availability. Understanding the interplay of environmental factors (such as resource density and its distribution) with biological factors (such as growth rate, perceptual radius, and metabolic costs) allows the exploration of possible trajectories by which perception may evolve. We used a complex systems perspective by employing an agent-based model in lieu of alternative approaches involving deterministic equations. We incorporated a context-dependent movement strategy for each agent where it switches between undirected (random walk) and directed (advective) movement based on its perception of resources. To supply additional biological realism, we investigated evolution in a reproductive context, imposing limits on the amount of resources an individual can gather and store and exploring a wide range of initial conditions and parametric scenarios.

Focusing on the evolved distribution of perceptual radius, we observed a nonlinear, non-monotonic response as a function of resource density. We found that the distribution of perceptual radii as a function of resources quickly converged to a sharp peak and then increased in variance. Resources play a major role in determining the stability of equilibria of the system, controlling whether or not perceptual ranges emerge at all. In addition, we found that the system’s behavior mirrored some biological aspects, with evolved perceptual abilities depending on metabolic and energetic costs.


## Short Description of the files in this repository ##

**Base scripts** 
- `avg_sweeper.py`: Python wrapper code for easy parameter sweeps of `capped_energy_resc_refill.go`. The `Simulation` class is initialized by a given parameter sweep's parameters and contains all the methods needed to run, evaluate, and visualize the sweep. Sample workflow for all the plots in the paper is given in the code at the end. The class can be made to inherit from `Thread` for multithreading (currently not implemented, however). 
- `capped_energy_resc_refill_v2.go`: the base Go code to run the simulations. This is called by `avg_sweeper.py` repeatedly in parameter sweeps. Must be compiled locally before use; `avg_sweeper.py` expects the executable to be called `capped_energy_resc_refill.out` but this can be easily changed in the code. The setup is explained more in the paper.
- `evolution_sim.py`: Python wrapper code for easy parameter sweeps of `vision_2pops.go`, similar to `avg_sweeper.py`. The functions after the `Simulation` class were written before the class was made, so the class simply invokes the functions rather than duplicating the code. Sample workflow is shown in the code at the end, along with some visualization code for a figure from the paper.
- `vision_2pops.go`: the base Go code to run the two-population evolution verification simulations. This is called by `evolution_sim.py` repeatedly in the parameter sweeps. Must be compiled locally before use; `evolution_sim.py` expects the executable to be called `evolution_sim.out` but this can be easily changed in the code. The setup for these sweeps is explained more in the paper.
- `sweeper.sh`: deprecated Bash code for parameter sweeps from earlier development stages.
- `vision.go`, `vision_09182019.go`: deprecated Go code from earlier development stages.
