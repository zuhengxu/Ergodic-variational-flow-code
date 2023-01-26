library(dplyr)

agg_sum <- function(name) {
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
  
  df = data.frame(min(df_final$`AVG. KSD`), mean(df_final$`AVG. KSD`), 
                  sd(df_final$`AVG. KSD`), sum(df_final$`NAN RATIO` > 0))
  df = round(df, 2)
  colnames(df) = c("MIN. KSD", "AVG. KSD", "SD. KSD", "#FAIL")
  rownames(df) = c(name)
  
  return(df)
}

##########
# banana
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//banana//result")
banana = agg_sum("banana")

##########
# cross
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//cross//result")
cross = agg_sum("cross")

##########
# heavy_reg
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//heavy_reg//result")
heavy_reg = agg_sum("t regression")

##########
# lin_reg_heavy
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//lin_reg_heavy//result")
lin_reg_heavy = agg_sum("linear regression (heavy tail)")

##########
# linear_regression
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//linear_regression//result")
linear_regression = agg_sum("linear regression")

##########
# logistic_reg
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//logistic_reg//result")
logistic_regression = agg_sum("logistic regression")

##########
# neals_funnel
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//neals_funnel//result")
neals_funnel = agg_sum("Neal's funnel")

##########
# poiss
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//poiss//result")
poiss = agg_sum("Poisson regression")

##########
# sparse_regression
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//sparse_regression//result")
sparse_regression = agg_sum("sparse regression")

##########
# warped_gaussian
##########
setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example//warped_gaussian//result")
warped_gaussian = agg_sum("warped Gaussian")

df_final = rbind(banana, cross, neals_funnel, warped_gaussian, 
                 linear_regression, lin_reg_heavy, logistic_regression, poiss,
                 heavy_reg, sparse_regression)

setwd("C://projects//ErgFlow//Ergodic-variational-flow-code//example")
write.csv(df_final, "aggregated.csv")