library(dplyr)
library(ggplot2)
library("tidyr")
library(ggpubr)
library(corrplot)
library(latex2exp)

rm(list=ls())

root_folder = "SingleObjectiveExperiments"
#folders = c("ackley2", "ackley4", "ackley8_nogrid", "ackley8_N10", "ackley8")
folders = c("ackley2", "ackley3", "ackley4", "ackley8")

first <- TRUE
for (exp in folders) {
  files <- list.files(path = paste(root_folder, exp, sep = "/"))
  for (f in files)
  {
    print(f)
    if (grepl("csv", f, fixed = TRUE)) {
      if (first) {
        df <- read.csv(paste(root_folder, exp, f, sep = "/"))
        df$file = f
        df$testF = exp
        first <- FALSE
      }
      else {
        df_temp <- read.csv(paste(root_folder, exp, f, sep = "/"))
        df_temp$file = f
        df_temp$testF = exp
        df <- rbind(df, df_temp)
      }
    }
  }  
}

summary(df)

df <- df %>% 
  select(-X) %>%
  mutate(time = abs(time)) %>%
  mutate(file = as.factor(file)) %>%
  mutate(testF=as.factor(testF)) %>%
  mutate(acqF=as.factor(acqF)) %>%
  mutate(exp_id=as.factor(exp_id))

summary(df)

metrics <- c("r_t")


for (folder in folders) {
  for (m in metrics) {
    
    plot_object <- df %>%
      filter(testF==folder) %>%
      select(exp_id, acqF, ns, r_t, R_t) %>%
      gather(key = "metric", value = "measure", 4:5) %>%
      group_by(acqF, ns, metric) %>%
      summarise(measure = mean(measure, na.rm=TRUE)) %>%
      mutate(metric = as.factor(metric)) %>%
      filter(metric == m) %>%
      ggplot(aes(x=ns, y = measure, col = acqF)) +
      geom_line() + ggtitle(paste(folder, m, " "))
    
    print(plot_object)
  }
}


acqs = c("ei", "pi", "simulated_mes5")

for (folder in c("ackley")) {
  for (a in acqs){
    print(paste(folder, a, m, sep=" - "))
    
    plot_object <-  df %>%
        filter(testF==folder)  %>%
        filter(acqF==a) %>%
        ggplot(aes(x=ns, y = r_t, col=exp_id)) +
        geom_line() + ggtitle(paste(folder, a, m, " "))
    
    print(plot_object)

  }
}

for (folder in folders) {
    plot_time <- df %>%
      filter(testF==folder) %>%
      select(acqF, time) %>%
      ggplot(aes(x=acqF, y=time)) +
      geom_violin() +
      scale_y_log10() +
      ggtitle(folder)
      
    print(plot_time)
}
