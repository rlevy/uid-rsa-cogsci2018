library(tidyverse)

dat <- read.table("results.txt",header=T)
dat$stable <- with(dat,thatrate > 0.001 & thatrate < 0.999)
dat <- subset(dat, ! (k==1.0 & c==0.0))
dat.summary <- dat %>% group_by(k,c) %>%
  dplyr:::summarise(stable=mean(stable),r=mean(r))

## stable optionality plot
ggplot(dat.summary,aes(k,c)) + geom_tile(aes(fill=stable),colour="white") +
  labs(y=expression(paste("String length cost parameter ", c)),
       fill="stable\noptionality\nrate") +
  theme_classic() +
  scale_x_continuous(name=expression(paste("Nonuniformity penalization parameter ",k)),breaks=seq(1,2,by=0.2))

## Distribution of marginal frequencies of optional marker t at fixed points with stable optionality
ggplot(subset(dat,stable),aes(x=thatrate,y=..density..)) + geom_histogram(bins=42) + scale_x_continuous(limits=c(-0.05,1.05)) +
  ylab("Probability density") +
  xlab(expression(paste("Marginal frequency of optional marker ",t))) +
  theme_classic()

## distribution of correlations between phrase onset probability and t-rate at fixed points with stable optionality
ggplot(subset(dat,stable),aes(x=r,y=..density..)) + geom_histogram(bins=42) + scale_x_continuous(limits=c(-1.05,1.05)) +
  ylab("Probability density") +
  xlab("Pearson correlation between\nphrase onset & t probabilities") + 
  theme_classic()
