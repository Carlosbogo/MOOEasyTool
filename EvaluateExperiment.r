library(dplyr)
library(ggplot2)
library("tidyr")
library(ggpubr)

rm(list=ls())

exp = "MOOackley/"

files <- list.files(path = exp)
first <- TRUE
for (f in files)
{
  if (grepl("csv", f, fixed = TRUE)) {
    if (first) {
      df <- read.csv(paste(exp,f, sep = ""))
      df$name <- f
      first <- FALSE
    }
    else {
      df_temp <- read.csv(paste(exp,f, sep = ""))
      df_temp$name <- f
      df <- rbind(df, df_temp)
    }
  }
}

head(df)

df <- df %>% 
  select(-X) %>%
  mutate(name=as.factor(name)) %>%
  mutate(
    name = recode(name,
                  basic_mes_acq.csv = "Basic MES",
                  mes_acq.csv = "MES",
                  mesmo_acq.csv ="MESMO",
                  random_acq.csv = "RANDOM"
    )
  ) %>%
  relocate(name, .before="ns")%>%
  mutate(ds = d1+d2) %>%
  relocate(ds, .after="d2")

head(df)

df %>% group_by(name, ns) %>% summarise(N=n()) %>% filter(ns==min(ns))

ns_max <- 100

######################
## Distance metrics ##
######################
df_metrics <- df %>%
  gather(key="type", value="d", c(4:8)) %>%
  group_by(ns, type, name) %>%  
  summarise(dev = sd(d), distance=mean(d), log_d=mean(log10(d)), log_dev=sd(log10(d)))

gd <- df_metrics %>%
  filter(type=='d', ns<=ns_max) %>%  
  ggplot(aes(x=ns, y=distance, color=name)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=distance*10^log_dev, ymax=distance*10^-log_dev), width=.6) +
  ylab("Hausdorff Distance") + xlab("Iteration Number") +
  scale_y_log10()

gd1 <- df_metrics %>%
  filter(type=='d1', ns<=ns_max) %>%  
  ggplot(aes(x=ns, y=distance, color=name)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=distance*10^log_dev, ymax=distance*10^-log_dev), width=.6) +
  ylab("Directed Hausdorff Distance (Estimated-Real)") + xlab("Iteration Number") +
  scale_y_log10()

gd2 <- df_metrics %>%
  filter(type=='d2', ns<=ns_max) %>%  
  ggplot(aes(x=ns, y=distance, color=name)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=distance*10^log_dev, ymax=distance*10^-log_dev), width=.6) +
  ylab("Directed Hausdorff Distance (Real-Estimated)") + xlab("Iteration Number") +
  scale_y_log10()

gds <-df_metrics %>%
  filter(type=='ds', ns<=ns_max) %>%  
  ggplot(aes(x=ns, y=distance, color=name)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=distance*10^log_dev, ymax=distance*10^-log_dev), width=.6) +
  ylab("Addittion of Directed Hausdorff Distances") + xlab("Iteration Number") +
  scale_y_log10()

ggarrange(gd, gd1, gd2, gds)

######################
#### HVs  metrics ####
######################

df_metrics %>%
  filter(type=='hp', ns<=ns_max) %>%  
  ggplot(aes(x=ns, y=distance, color=name)) +
  geom_line() + 
  #geom_errorbar(aes(ymin=distance*10^log_dev, ymax=distance*10^-log_dev), width=.2) +
  ylab("Hypervolume diference") + xlab("Iteration Number") +
  scale_y_log10()

df %>%
  ggplot(aes(x=name, y=time)) + 
  geom_boxplot() + ylab("Time (s)") + xlab("Acquisition Function") +
  scale_y_log10()

df %>% select(name, time) %>% filter(!is.na(time)) %>% group_by(name) %>% summarise(time = mean(time))

df %>% filter(name=="RANDOM") %>% 
  ggplot(aes(x=ns, y=hp, color=idexp, group=idexp)) + 
  geom_line() + scale_y_log10()

######################
#### Correlations ####
######################
head(df)
df_cors <- df %>% 
  group_by(name, idexp) %>% 
  summarise(c_d_d1 = cor(d,d1),
            c_d_d2 = cor(d,d2),
            c_d_ds = cor(d,ds),
            c_hp_d  = cor(hp,d),
            c_hp_d1 = cor(hp,d1),
            c_hp_d2 = cor(hp,d2),
            c_hp_ds = cor(hp,ds))

p_c_hp_d <- df_cors %>%
  ggplot(aes(x=name, y=c_hp_d)) +geom_boxplot()

p_c_hp_d1 <- df_cors %>%
  ggplot(aes(x=name, y=c_hp_d1)) +geom_boxplot()

p_c_hp_d2 <- df_cors %>%
  ggplot(aes(x=name, y=c_hp_d2)) +geom_boxplot()

p_c_hp_ds <- df_cors %>%
  ggplot(aes(x=name, y=c_hp_ds)) +geom_boxplot()

ggarrange(p_c_hp_d, p_c_hp_d1, p_c_hp_d2, p_c_hp_ds)

head(df)

df %>% group_by(name, ns) %>%
  summarise(d = mean(d), d1 = mean(d1), d2 = mean(d2), ds=d1+d2, hp=mean(hp)) %>%
  group_by(name) %>%
  summarise(cor_d_hp = cor(d,hp),
            cor_d1_hp = cor(d1,hp),
            cor_d2_hp = cor(d2,hp),
            cor_ds_hp = cor(ds,hp),
            cor_d_d1 = cor(d,d1),
            cor_d_d2 = cor(d,d2),
            cor_d_ds = cor(d,ds))
