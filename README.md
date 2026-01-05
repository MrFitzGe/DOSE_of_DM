# DOSE_of_DM
[DOSE methodolgy](https://www.nber.org/papers/w33013) for preference elicitation using the [AX](https://ax.dev/) platform by Meta.


### Why DOSE?
* allows participants to make mistakes and learn over time
* measures the information gain of the next trial, which allows for early stopping - reward participants with clear preferences
* elicits preferences quickly! - half the error rate or 50% fewer trials needed (on average)
* reduces the measurement error of the experiment


### How does it work
Call-and-reponse interaction with a Bayesian optimization algorithm (Gaussian Process). 
Essentially there are a few 'burn-in' trials, say N=10 to get a lay of the land, and then the GP will start to explore the parameter space and exploit areas that have shown real value. The algorithm generates 'next up' parameter set, we pass those parameters to the participant as stimuli (#TODO is this from a set list of questions or dynamically generated with an LLM or just string-interpolation?) and collect a response. We pass that response back to the algorithm and it handles the update process. 

The algorithm optimizes information gain towards a participant's indiference point so it generates the next best set of parameters to find that indiference point - i.e. it looks for tough choices **and** coverage of the search space.


### Further Reading

* [Unexpected Improvements to Expected Improvement
for Bayesian Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/file/419f72cbd568ad62183f8132a3605a2a-Paper-Conference.pdf)


## Subjective Value computational models
* loss and risk aversion via prospect theory (Kahneman & Tversky)
* hyperbolic model used to represent temporal discounting (Ainslie; Mazur 1975)
* probability discounting model (Green and Myerson)
* sigmoidal model of effort discounting (Klein-Flugge et al. 2015)
* power model of effort discounintg (Hartmann, Hager, Tobler, and Kaiser 2013; Wojciech Białaszek ,Przemysław Marcowski,Paweł Ostaszewski 2017)
