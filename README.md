# uid-rsa-cogsci2018

This repo contains code to replicate the simulations presented at:

Roger P. Levy. 2018. Communicative Efficiency, Uniform Information
Density, and the Rational Speech Act Theory.  Proceedings of the 40th
Annual Meeting of the Cognitive Science Society.

The script rsa_speaker.py is Python 2.7 and can be run as follows to
obtain N replicates for various values of UID cost parameter k and
utterance length cost parameter c:

python rsa_speaker --start_seed I --num_seeds N --num_processes P > results.txt

As the code was set up at the time of CogSci camera ready submission
on May 14, there is no use in setting P>21, but if you have fewer than
21 cores available you may want to set P to a lower integer value.

You can then generate the graphs in the paper by running the R code in analyze_results.R.
