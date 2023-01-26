library(dplyr)

agg_sum <- function() {
  df = read.table("neo_fix.csv", header = TRUE)
  # Create a new column "group" 
  df$group <- rep((1:(nrow(df)/3)),each =3)
  # group by group and take the mean
  df_mean <- df %>% group_by(group) %>% summarise_all(mean)
  df_a = read.table("neo_adp.csv", header = TRUE)
  # Create a new column "group" 
  df_a$group <- rep((1:(nrow(df_a)/3)),each =3)
  # group by group and take the mean
  df_a_mean <- df_a %>% group_by(group) %>% summarise_all(mean)
  df_final = rbind(df_mean, df_a_mean)
  df_final = df_final[,c(1,3,4,5,6,7,9)]
  df_final = round(df_final, 2)
  colnames(df_final) = c("group", "GAMMA", "#STEPS", "STEPSIZE", "#CHAINS", "AVG. KSD", "NAN RATIO")
  write.csv(df_final, "neo_table.csv")
}

##########
# banana
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//banana//result")
agg_sum()

##########
# cross
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//cross//result")
agg_sum()

##########
# heavy_reg
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//heavy_reg//result")
agg_sum()

##########
# lin_reg_heavy
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//lin_reg_heavy//result")
agg_sum()

##########
# linear_regression
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//linear_regression//result")
agg_sum()

##########
# logistic_reg
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//logistic_reg//result")
agg_sum()

##########
# neals_funnel
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//neals_funnel//result")
agg_sum()

##########
# poiss
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//poiss//result")
agg_sum()

##########
# sp_reg_big
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//sp_reg_big//result")
agg_sum()

##########
# sparse_regression
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//sparse_regression//result")
agg_sum()

##########
# warped_gaussian
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//warped_gaussian//result")
agg_sum()