# Introduction to Generative Adversarial Networks
- **Unsupervised** machine learning task: discover and learn regularities in input data in such a way that model can be used to generate or output new examples that could be plausible in original dataset
- Frame problem as supervised learning with two submodels: **generator** and **discriminator**
    - Generator is trained to generate new examples
    - Discriminator classifies examples as real or fake
    - Generator is updated via discriminator
    - Training is zero-sum game (adversarial), ends when discriminator model is fooled about half the time
## Discriminative vs. Generative Modeling
- Discriminative modeling = **classification**, discriminate examples of input variables across classes
- **Generative modeling**: unsupervised models that summarize distribution of input variables may be used to generate new examples in the input distribution
    - By sampling from distribution of inputs, it is possible to generate synthetic data points in the input space
## Generator Model
- Input is fixed-length vector drawn randomly from Gaussian distribution --> seed generative process
- After training, points in multi-dimensional vector space will correspond to points in domain --> compressed representation of data distribution
- **Latent variables** = projection or compression of a data distribution
    - Generator model applies meaning to points in a chosen latent space, s.t. new points drawn from latent space can be provided to generator model as input and used to generate new and different output examples
- Kept after training
- Can be repurposed, i.e., feature extraction layers can be used for transfer learning tasks
## Discriminator Model
- Takes example from input (real or generated) --> **binary classification** of real or fake/generated
    - Real example comes from training set
- Discarded after training
## Two-Player Game
- Unsupervised learning problem, but training process for generative model is framed as supervised
- Two models trained in parallel
    - Generator can be like a "counterfeiter", trying to make fake money, and discriminator is like the police, allowing real money but catching counterfeit money
    - Counterfeiter must learn to make money that is indistinguishable from real money
- **Zero-sum game**: On successful classification by discriminator, it is rewarded, and generator is penalized with large updates to model parameters, and vise versa
## GANs and Convolutional Neural Networks
- GANs typically work with image data and use CNNs as generator and discriminator
- Modeling image data --> latent space, input to generator, provides compressed representation of set of images or photographs used to train model, and generator generates new images
## Conditional GANs (cGANs)
- Extension of GAN --> conditionally generate output
- Generative model can be trained to generate new examples from input domain, where input (random vector from latent space) is provided (conditioned) with some additional input
    - Extra input can be class label (i.e., actual digit for digit classification)
    - Fed as input layer into both discriminator and generator
- Conditional GAN can then be used to generate images of a given type
- For imag-to-image translation, discriminator is provided examples of real and generated nighttime photos as well as (conditioned on) real daytime photos
    - Generator provided with a random vector from latent space as well as (conditioned on) real daytime photos
# Pix2Pix GANs
- A cGAN designed for general-purposed image-to-image translation
- Generation of output is conditional on input source image
- Discriminator provided both with a source image and target image and determines whether target image is plausible transformation of source one
- Generator trained with **adversarial loss**, updated with L1 loss, measured between generated image and expected output image 