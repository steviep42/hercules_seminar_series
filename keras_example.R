library(keras)

# See https://heartbeat.fritz.ai/binary-classification-using-keras-in-r-ef3d42202aaa

library(dplyr)
library(caret)
library(mlbench)
data(PimaIndiansDiabetes)
pm <- PimaIndiansDiabetes

idx <- createDataPartition(pm$diabetes,p=.8,times=1,list=FALSE)

train <- pm[idx,]
test <- pm[-idx,]

x_train <- train[,1:8]
y_train <- train[,9]

x_test <- test[,1:8]
y_test <- test[,9]

model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 12, activation = 'relu', kernel_initializer='RandomNormal', input_shape = c(8)) %>% 
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'linear')

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 50, 
  validation_split = 0.2
)

library(caret)
library(dplyr)
library(mlbench)

data("PimaIndiansDiabetes")

# Get a short name for the data frame
pm <- PimaIndiansDiabetes

# Do One Hot Encoding On The diabetes column
dummy_data <- fastDummies::dummy_cols(pm,remove_first_dummy = TRUE)

final <- dummy_data[,-9]
index <- createDataPartition(final$diabetes_pos, p=0.7, list=FALSE)

final.training <- final[index,]
final.test <- final[-index,]

X_train <- final.training %>% 
  select(-diabetes_pos) %>% 
  scale()

y_train <- to_categorical(final.training$diabetes_pos)

# 

X_test <- final.test %>% 
  select(-diabetes_pos) %>% 
  scale()

y_test <- to_categorical(final.test$diabetes_pos)

#
model <- keras_model_sequential() 

model %>% 
  layer_dense(units = 16, activation = 'relu', input_shape = ncol(X_train)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 8, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 4, activation = 'relu') %>%
  layer_dropout(rate = 0.1) %>%
  layer_dense(units = 2, activation = 'softmax')

history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

model %>% fit(
  X_train, y_train, 
  epochs = 100, 
  batch_size = 16,
  validation_split = 0.3,
  callbacks = callback_tensorboard("logs/run_1")
)
tensorboard("logs/run_1")


predictions <- model %>% predict_classes(X_test)
myt <- table(predictions,final.test$diabetes_pos) 
cat("Computed Accrucay on Test Data is: ",round(sum(diag(myt))/sum(myt),2),"\n")

acc <- model %>% evaluate(X_test, y_test)




## predict glucose levels

data(PimaIndiansDiabetes)
pm <- PimaIndiansDiabetes

# Do one hot encoding on the diabetes column
data <- fastDummies::dummy_cols(pm)

# Create a train / test pair
idx <- createDataPartition(data$glucose,p=.8,times=1,list=FALSE)

train <- data[idx,]
test <- data[-idx,]

# Keras requires the following

trainingtarget <- train[, 'glucose']
testtarget <- test[, 'glucose']

training <- train[,c(1,3:8,10:11)]
testing <- test[,c(1,3:8,10:11)]

# Do some scaling
m <- colMeans(training)
s <- apply(training, 2, sd)
training <- as.matrix(scale(training, center = m, scale = s))
testing <- as.matrix(scale(testing, center = m, scale = s))


# Create Model

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(training)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 1)

# Compile
model %>% compile(loss = 'mse',
                  optimizer = 'rmsprop',
                  metrics = 'mae')

# Fit Model
mymodel <- model %>%
  fit(training,
      trainingtarget,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)

# Evaluate
model %>% evaluate(testing, testtarget)
pred <- model %>% predict(testing)
mean((testtarget-pred)^2)
plot(testtarget, pred)


# Prepping for Keras has a number of steps which are a bit tedious
# 1 - load data
# 2 - if classification, you need to make the outcome in one hot encoding format
#     there are many ways to do this - too many in fact
# 3   Create a train / test pair
# 4   Separate the X and Y from both train and test - Keras wants this
# 5   turn the Y variables into categorical
# 6   Scale the predictor / X data and make sure they are matrices


data("PimaIndiansDiabetes")

# Get a short name for the data frame
pm <- PimaIndiansDiabetes

# Do One Hot Encoding On The diabetes column
dum_pm <- dummyVars(~.,pm,fullRank = TRUE)
final_pm <- data.frame(predict(dum_pm,pm))


index <- createDataPartition(final_pm$diabetes.pos, p=0.7, list=FALSE)

train <- final_pm[index,]
test <- final_pm[-index,]


# Keras requires the following

training_y <- train[, 'diabetes.pos']
training_y <- to_categorical(training_y,2)

testing_y <- test[, 'diabetes.pos']
testing_y <- to_categorical(testing_y,2)

training_x <- train[,1:8]
testing_x  <- test[,1:8]


training_x <- as.matrix(scale(training_x))
testing_x <- as.matrix(scale(testing_x))

# Create Model

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 128, activation = 'relu', input_shape = ncol(training_x)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 2)

model <- keras_model_sequential()
model %>% 
  layer_dense(units = 10, activation = 'relu', input_shape = ncol(training_x)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 2)

# Compile
history <- model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

# Fit Model
mymodel <- model %>%
  fit(training_x,
      training_y,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2,
      callbacks = callback_tensorboard("logs/run_1"))

tensorboard("logs/run_1")

plot(model)
