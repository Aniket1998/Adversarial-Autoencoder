# Adversarial Autoencoder

Experiments with Adversarial Autoencoder by Makhzani et. al. I'm primarily doing the following experiments
* Changes to the encoder decoder architecture
* Varying the adversarial loss from Minimax to Wasserstein, Least Squares, Energy Based and Boundary Equilibrium Losses
* Changing the reconstruction term to the kind used in Larsen et. al. (this gets cumbersome, there's two auxilliary discriminators now, the one that tells the images apart and the ones that tell the latent code apart)

## Network Architecture

Encoder: Step down convolutional encoder. Works on arbitrary image sizes whose dimensions are powers of 2. Step down is as follows
(n,n,ch) -> (n/2, n/2, d) -> (n/4, n/4, d*2) -> (1, 1, d*n/2)
Followed by 2 fully connected layers d * n/2 -> z for mean and variance of the inference model q(z|x) which is assumed to be a factored Gaussian

Decoder: Step up transposed convolutional decoder that is symmetric to the encoder
Except for 2 fully connected layers, there are two convolutional layers for the mean and variance of the conditional p(x|z) which is assumed to be a factored Gaussian
Alternative Architecture: Same as the previous architecture except the decoder model directory generates the images

## Results
[To be uploaded]
