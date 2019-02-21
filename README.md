Welcome to the comma.ai Programming Challenge!
======

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Deliverable
-----

Your deliverable is test.txt. E-mail it to givemeajob@comma.ai, or if you think you did particularly well, e-mail it to George.

Evaluation
-----

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.

Twitter
------

<a href="https://twitter.com/comma_ai">Follow us!</a>

###Ideas

Train the network normally, but use the same technique as adversarial autoencoder to shape the latent space just before prediction
Then, use our output from the discriminator and normalize it somehow and then generate the next picture using the speed

Train it with one frame from the future
Use an Autoencoder to predict the next frame for test file
