In the code FGSM is implemented using [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox).The toolbox works only with single output layer models and its classifier wrapper supports only Sequential() models when disabling eager execution. Nonetheless there's also an experimental wrapper that does support eager execution and keras.Model() models at the cost of efficiency. In this case, I found that working with Sequential() models  without disabling eager execution was cleaner and led to the same results. Furthermore I applied the attack only to the main input because of memory limits.   \
FGSM exploits the gradients of the network in order to create an image which maximises the loss. I does it by using the gradients of the loss wrt to the input image so to find the pixels which contribute the most and then adding a perturbation to those pixels. The perturbation has the same direction of the gradient but its magnitude depends on a small value ε.




\
[1] Ian J. Goodfellow, Jonathon Shlens, Christian Szegedy. In *Explaining and Harnessing Adversarial Examples*, 2014. https://arxiv.org/abs/1412.6572
