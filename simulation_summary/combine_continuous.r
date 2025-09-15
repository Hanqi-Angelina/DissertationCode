library(tidyverse)
library(readr)
df_gaussian = read_csv("evaluation_summary_1to80_gaussian.csv")
df_gaussian_r = read_csv("evaluation_summary_80_gaussian_r.csv")
df_laplace = read_csv("evaluation_summary_1to80_laplace.csv")
df_laplace_r = read_csv("evaluation_summary_80_laplace_r.csv")
df_threshold = read_csv("evaluation_summary_all_dedup.csv")
df_gaussian_all = rbind(df_gaussian, df_gaussian_r)
df_laplace_all = rbind(df_laplace, df_laplace_r)
rm(df_gaussian)
rm(df_gaussian_r)
rm(df_laplace)
rm(df_laplace_r)

laplace_scores = df_laplace_all%>%
  group_by(Scenario, Method, Sample_Size)%>%
  summarise(n_eff = n_distinct(Run_ID), 
            mean_f1 = round(mean(`F1 Score`),2), 
            sd_f1 = round(sd(`F1 Score`),2),
            mean_shd = round(mean(SHD), 2), 
            sd_shd = round(sd(SHD), 2))
gaussian_scores = df_gaussian_all%>%
  group_by(Scenario, Method, Sample_Size)%>%
  summarise(n_eff = n_distinct(Run_ID), 
            mean_f1 = round(mean(`F1 Score`),2), 
            sd_f1 = round(sd(`F1 Score`),2),
            mean_shd = round(mean(SHD), 2), 
            sd_shd = round(sd(SHD), 2))
threshold_scores = df_threshold%>%
  group_by(Scenario, Method, Threshold)%>%
  summarise(n_eff = n_distinct(Run_ID), 
            mean_f1 = round(mean(`F1 Score`),2), 
            sd_f1 = round(sd(`F1 Score`),2),
            mean_shd = round(mean(SHD), 2), 
            sd_shd = round(sd(SHD), 2))
threshold_scores = threshold_scores%>%
  mutate(Threshold = factor(Threshold, levels = c("symmetric", "mild", "moderate")))%>%
  arrange(Threshold)