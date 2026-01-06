# Rolladex
![Rolladex logo](https://github.com/MrFitzGe/DOSE_of_DM/blob/main/Rolladex_logo.png?raw=true)

## Adaptive Decision Making Experiments
Common methodology for eliciting decision preferences in participants using repeated choice task trials with varying parameters. This repo was inspired by the
[DOSE methodolgy](https://www.nber.org/papers/w33013) for adaptive preference elicitation but uses the powerful [AX](https://ax.dev/) platform by Meta underneath the hood. This combination has several advantages:
* conduct experiments with fewer and more quantifiably meaningful trials (the DOSE paper claims ~50% fewer trials needed to achieve stable and reliable model parameter estimates)
* enable early stopping to save participants time and reward consistent choices
* fast bayesian optimization thanks to [BoTorch](https://botorch.org/) which also scales well if GPUs are needed for complex designs
* extensible platform building from Ax and BoTorch which are interoperable with scipy and pytorch methods already existing
* quantifies the measurement error of the experiment



### How does it work
Call-and-reponse interaction with a Bayesian optimization algorithm (Gaussian Process model using Expected information Gain). 
Essentially there are a few 'burn-in' trials (~10) to get a lay of the land - these could also be instructional/practice trials, and then the GP will start to explore the parameter space and exploit values of reward and cost that provide meaningful information. The algorithm generates the 'next up' set of stimuli parameters, we show those parameters to the participant as stimuli (#TODO LLM generated wording of the stimuli?) and collect a choice response. We take the choice and calculate the log-likelihood and entropy of our computational model fit, pass that information back to the GP and it automatically updates and returns the next step (either another set of trial stimuli or the suggestion to end the experiment). 

The algorithm optimizes information gain towards a participant's indiference point so it generates the next best set of parameters to reduce entropy - i.e. it looks for difficult choices **and** coverage of the search space **and** inconsistent choices.


### Further Reading

* [Unexpected Improvements to Expected Improvement
for Bayesian Optimization](https://proceedings.neurips.cc/paper_files/paper/2023/file/419f72cbd568ad62183f8132a3605a2a-Paper-Conference.pdf)


## Subjective Value computational models
### Uncertainty
* loss and risk aversion via prospect theory (Kahneman & Tversky)
* probability discounting model (Myerson & Green)
### Temporal discounting
* hyperbolic model used to represent temporal discounting (Ainslie; Mazur 1975)
### Effort discounting
* sigmoidal model of effort discounting (Klein-Flugge et al. 2015)
* power model of effort discounintg (Hartmann, Hager, Tobler, and Kaiser 2013; Wojciech Białaszek, Przemysław Marcowski, Paweł Ostaszewski 2017)


## RoadMap TODO list
* [ ] Past choice memory + entropy calculation outside of the GP == how should the interface work between participant / experimenter / Ax client?
* [ ] Early stopping alerts (what is the threshold? how do we determine this)
* [ ] Add other models than just hyperbolic discounting and have an organized system for model selection
* [ ] Ax analyses and plots
* [ ] Set up the experiment = interface for the experimenter and the hyperparameters like number of burn in trials, how to set them, model selection, number of parameters (need to update AIC calcs), etc.
