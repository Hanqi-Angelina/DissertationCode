library(tidyverse)
library(readr)
library(igraph)
library(ggrepel)

bootstrap_data = read_csv("Bootstrap/summary_prop.csv")[c(-1)]
bootstrap_sum = read_csv("Bootstrap/sum_new.csv")[c(-1)]
subsample_data = read_csv("Subsampling/summary_prop.csv")[c(-1)]
subsample_sum = read_csv("Subsampling/sum_new.csv")[c(-1)]
# ====== bootstrap results ============ #
bootstrap_sum%>%
  group_by(Error)%>%
  summarise(Count = n(), mean_SHD = mean(SHD), SD_SHD = sd(SHD), 
            mean_latents = mean(n_latents), sd_latents = sd(n_latents))

mean_bootstrap_results = bootstrap_data%>%
  group_by(u,v)%>%
  mutate(prop = count/sum(count), sum = sum(count))

mean_bootstrap_results = mean_bootstrap_results%>%
  filter(((edge_type == "u->v")|(edge_type == "v->u"))|(edge_type == "undirected"))%>%
  group_by(u, v)%>%
  mutate(undirected_props = sum(prop))%>%
  arrange(desc(prop))

# ====== subsampling results =========== #
subsample_sum%>%
  group_by(Error)%>%
  summarise(Count = n(), mean_SHD = mean(SHD), SD_SHD = sd(SHD), 
            mean_latents = mean(n_latents), sd_latents = sd(n_latents))
mean_subsample_results = subsample_data%>%
  group_by(u,v)%>%
  mutate(prop = count/sum(count), sum = sum(count))

mean_subsample_results = mean_subsample_results%>%
  filter(((edge_type == "u->v")|(edge_type == "v->u"))|(edge_type == "undirected"))%>%
  group_by(u, v)%>%
  mutate(undirected_props = sum(prop))%>%
  arrange(desc(prop))

# write_csv(mean_subsample_results, "MainResults/mean_subsample_results.csv")

# for clean skewness display
ordinal_data = read_csv("../data/original_data.csv")
ordinal_main = ordinal_data%>%
  select(starts_with("Worry_")|starts_with("Sleep_")|starts_with("MistrustB_"))

library(moments)
skew1 = lapply(ordinal_main, skewness)%>%
  as_tibble()%>%
  pivot_longer(cols = c(1:27), names_to = "Name", values_to = "Skewness")

skew1%>%
  separate(Name, into = c("Construct", "Item"), sep = "_")%>%
  group_by(Construct)%>%
  summarise(construct_mean = round(mean(Skewness),2), construct_sd = round(sd(Skewness), 2))

skew1 = skew1%>%
  mutate(Name = str_replace(Name, "_",""), Skewness = round(Skewness,2))

kurt1 = lapply(ordinal_main, kurtosis)%>%
  as_tibble()%>%
  pivot_longer(cols = c(1:27), names_to = "Name", values_to = "Kurtosis")


