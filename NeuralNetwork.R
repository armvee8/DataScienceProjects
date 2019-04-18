library(neuralnet)
library(dplyr)
phish <- read.csv('/students/home/vweiss/Desktop/TrainingDataset.csv')

#Preprocessing the data for train
phish.train.NN <- phish.train
phish.train.NN$Result <- gsub(-1, 0, phish.train$Result)
phish.train.NN$Result <- as.numeric(phish.train.NN$Result)

#Preprocessing data for test
phish.test.NN <- phish.test
phish.test.NN$Result <- gsub(-1, 0, phish.test$Result)
phish.test.NN$Result <- as.numeric(phish.test.NN$Result)

nn.five <- neuralnet(Result ~ SSLfinal_State + URL_of_Anchor + web_traffic + having_Sub_Domain + Links_in_tags, data=phish.train.NN, hidden=c(3,4), algorithm = 'backprop', linear.output=FALSE, threshold=0.05)
nn$result.matrix
plot(nn.five)

#Variables of interest: 
#SSLfinal_State, URL_of_Anchor, web_traffic, 
#having_Sub_Domain, Links_in_tags, Result

#testing the resulting output
temp_test <- subset(phish.test.NN, select = c("SSLfinal_State","URL_of_Anchor", "web_traffic", "having_Sub_Domain", "Links_in_tags"))
nn.results <- compute(nn.five, temp_test)
results <- data.frame(actual = phish.test.NN$Result, prediction = nn.results$net.result)

#Making both the items in the confusion matrix factors

nn.results$net.result <- round(nn.results$net.result)
nn.results$net.result <- as.factor(nn.results$net.result)
phish.test.NN$Result <- as.factor(phish.test.NN$Result)


phish.ConfusionMatrix <- confusionMatrix(nn.results$net.result, phish.test.NN$Result)
print(phish.ConfusionMatrix)
