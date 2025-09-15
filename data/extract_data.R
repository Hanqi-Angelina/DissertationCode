# simple cleaning code provided together with the dataset
library(tidyverse)

source("correct_coding.R") #source the file with helper functions

dat2 <- foreign::read.spss("Oxford Trust Survey_FINAL DATA_adj2.sav")  # insert link to file
dat2 <- as_tibble(dat2)
dat2s <- dat2 %>% 
  select(RGPTSB:defences2)

dat2 %>% select(where(~is.ordered(.)))
sapply(dat2, class)

dat2sl <- dat2s %>% pivot_longer(everything())

dat2 <- dat2 %>% 
  mutate(across(where(~is.factor(.)), correct_coding)) %>%
  mutate(across(Q74_1:Q75_10, ~(`levels<-`(., 0:4))))

dat2 <- dat2 %>% 
  mutate(across(Q72_1:Q90_3, as.ordered)) %>%
  mutate(across(Q93_1:Q106, as.ordered)) %>%
  mutate(across(starts_with("Q108_") | starts_with("Q111_"), as.ordered))

ox_trust <- dat2
rm(dat2)

ox_trust_sum <- dat2s

# usethis::use_data(ox_trust)
# usethis::use_data(ox_trust_sum)

## factor analysis (structure) 


