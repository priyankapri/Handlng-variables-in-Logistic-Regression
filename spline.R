library(haven)
Splineregressiondata_Normalized_12_26_2020 <- read_dta("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/MSEEL_NormalizedN5335.dta")
View(Splineregressiondata_Normalized_12_26_2020)

data2<-data.frame(Splineregressiondata_Normalized_12_26_2020)
data = subset(data2, select = -c(wellstage) )
nrow(data) #5353

install.packages("rsample", "ggplot2", "earth", "caret", "vip","pdp")
library(rsample)   # data splitting 
library(ggplot2)   # plotting
library(earth)     # fit MARS models
library(caret)     # automating the tuning process
library(vip)       # variable importance
library(pdp)       # variable relationships


# Split the data into training and test set
set.seed(123)
training.samples <- data$gasproduction %>%
  createDataPartition(p = 0.75, list = FALSE)
train.data  <- data[training.samples, ]
test.data <- data[-training.samples, ]


# Predictor variables
x <- model.matrix(gasproduction~., train.data)[,-1]
# Outcome variable
y <- train.data$gasproduction



#Basic model

mars1<-earth
mars1 <- earth(
  gasproduction ~ .,  
  data = train.data 
)

print(mars1)

summary(mars1)

mars1_coef<- data.frame(mars1$coefficients)
View(mars1_coef)
write.csv(mars1_coef,"Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/MARSDeg1Coef.csv", row.names = TRUE)


tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/Mars1_Basic.tiff", units="in", width=6, height=5, res=300)
plot(mars1, which = 1)
dev.off()
vi(mars1)
backward <- step(mars1, direction = "backward", trace = 0)
vi(backward)
# Extract VI scores
vi(backward)

VIPNew <- vip(backward, num_features = length(coef(backward)), 
          geom = "point", horizontal = FALSE)

grid.arrange(VIPNew, nrow = 1)

# variable importance plots
tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/Mars1_vip_Basic.tiff", units="in", width=6, height=5, res=300)
p1 <- vip(mars1, num_features = 40) + ggtitle("Basic Mars")
gridExtra::grid.arrange(p1, ncol = 1)
dev.off()

modF<- lm(gasproduction~lpldcount+seismogenicindex+cumulativemoment+fracturecount+microseismicvolume+fluidvolume+bvalue+averagepressure+sourceradius+poissons_ratio+diffusivity+brittleness, train.data)
distPred <- predict(modF, test.data) 

summary(modF)

actuals_preds <- data.frame(cbind(actuals=test.data$gasproduction, predicteds=distPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)  # 85.5%


min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
# => 84.8%, min_max accuracy
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  
# => 21.79%, mean absolute percentage deviation


#DEgree2: Assess interaction as multicollineraity exists in the dataset

library(glmnet)

mars2 <- earth(
  gasproduction ~ .,  
  data = train.data, 
  degree=2
)

mars2_coef<- data.frame(mars2$coefficients)
write.csv(mars2_coef,"Box/Personal/Green Card/Abhash Data/ML Dec26th/MARSDeg2Coef.csv", row.names = TRUE)


print(mars2)
tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/Mars2_Basic.tiff", units="in", width=6, height=5, res=300)
plot(mars2, which = 1)
dev.off()

#Grid search

hyper_grid <- expand.grid(
  degree = 1, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)

head(hyper_grid)



# for reproducibiity
set.seed(123)

# cross validated model
tuned_mars <- train(
  x = subset(train.data, select = -gasproduction),
  y = train.data$gasproduction,
  method = "earth",
  metric = "RMSE",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = hyper_grid
)

# best model
tuned_mars$bestTune
##    nprune degree
## 23     23      1


tuned_mars$results %>%
  filter(nprune == tuned_mars$bestTune$nprune, degree == tuned_mars$bestTune$degree)

# degree nprune        RMSE  Rsquared         MAE      RMSESD   RsquaredSD       MAESD
# 1      1     23 0.006197611 0.9999748 0.003588107 0.004719942 4.939965e-05 0.002321679
print(tuned_mars)
# plot results
tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/CVMARS_RMSE.tiff", units="in", width=6, height=5, res=300)
ggplot(tuned_mars)
dev.off()

CVMARS_coef<- tuned_mars$finalModel$coefficients
View(CVMARS_coef)
write.csv(CVMARS_coef,"Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/CVMARS.csv", row.names = TRUE)

library(pdp)


# variable importance plots
tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/TunedMars_Basic.tiff", units="in", width=6, height=5, res=300)
p1 <- vip(tuned_mars, num_features = 40, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(tuned_mars, num_features = 40, geom = "point", value = "rss") + ggtitle("RSS")
gridExtra::grid.arrange(p1, p2, ncol = 2)
dev.off()


# Construct partial dependence plots
p1 <- partial(tuned_mars, pred.var = "lpldcount", grid.resolution = 10) %>% 
  autoplot()
p2 <- partial(tuned_mars, pred.var = "seismogenicindex", grid.resolution = 10) %>% 
  autoplot()
#p3 <- partial(tuned_mars, pred.var = c("lpldcount", "seismogenicindex"), 
 #             grid.resolution = 10) %>% 
  plotPartial(levelplot = FALSE, zlab = "yhat", drape = TRUE, colorkey = TRUE, 
              screen = list(z = -20, x = -60))

# Display plots side by side
tiff("Box/Personal/Green Card/Abhash Data/ML Dec26th/MARS/TunedMars_pdpplots.tiff", units="in", width=6, height=5, res=300)
gridExtra::grid.arrange(p1, p2, ncol = 2)
dev.off()


















mars2_coef<- data.frame(mars2$coefficients)
View(mars2_coef)






