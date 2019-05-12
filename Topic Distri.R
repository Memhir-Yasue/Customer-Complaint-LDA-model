library(tidyverse) # metapackage with lots of helpful functions
library(ggmap)
library(mapdata)
library(ggplot2)
library(ggthemes)
library(plotly)
library(gridExtra)
library(grid)

df_stat_Resturant <- read.csv('CIS3715/stats100.csv')
df_stat_Topic <- df_stat_Resturant %>%
  select(topic.0:topic.29)

colnames(df_stat_Topic) <- c(0:29)

# Average topic distribution
ggplot(stack(df_stat_Topic), aes(x=ind, y = values)) +
  geom_boxplot(outlier.shape = NA) + 
  xlab("Topics") +
  ylab("Avrg Probability by Business") +
  theme_economist()




# This is the raw topic distribtion by review

df_stat_ALL = read.csv('CIS3715/FINALL_doc_topic.csv')
df_stat_ALL_m <- df_stat_ALL %>%
  select(business_id,review,topic.0:topic.29)
colnames(df_stat_ALL_m) <- c('business_id','review', 0:29)


df_wait_examples <- df_stat_ALL %>%
  select(review,topic.27) %>%
  arrange(desc(topic.27))

d <- head(df_wait_examples)
colnames(d) <- c('Review','Topic 27 (table wait time) Avrg Probability Distribution')
grid.table(d)
write.csv(d, "27.csv")

# Focusing on that burger place...
df_stat_Burgerbar <- df_stat_ALL %>%
  filter(business_id == 'Cni2l-VKG_pdospJ6xliXQ') %>%
  select(topic.0:topic.29)

ggplot(stack(df_stat_Burgerbar), aes(x=ind, y = values)) +
  geom_boxplot(outlier.shape = NA) + 
  xlab("Topics") +
  ylab("Avrg Probability by Business") +
  theme_economist()



# This is a summary by resturant (100 x 34)
df_pizza <- df_stat_Resturant %>%
  select(name,category, topic.0:topic.29) %>%
  filter(str_detect(category,"Pizza"))

df_high_waiting_time <- df_stat_Resturant %>%
  select(name,topic.27) %>%
  arrange(desc(topic.27))

d <- head(df_high_waiting_time)
colnames(d) <- c('Resturant Name','Topic 27 (table wait time) Avrg Probability Distribution')
grid.table(d)


# fried stuff related issue
df_fried_stuff <- df_stat_Resturant %>%
  select(name,topic.18) %>%
  arrange(desc(topic.18))

d <- head(df_fried_stuff)
colnames(d) <- c('Resturant Name','Topic 18 (Fried stuff), Avrg Probability Distribution')
grid.table(d)


# front desk related issue
df_hotel_fd <- df_stat_Resturant %>%
  select(name,topic.14) %>%
  arrange(desc(topic.14))

d <- head(df_hotel_fd)
colnames(d) <- c('Business Name','Topic 14 (Front Desk), Avrg Probability Distribution')
grid.table(d)

