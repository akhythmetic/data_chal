library(dplyr)
library(ggplot2)
library(ROCR)

train <- read.csv("C:/Users/User/Desktop/Cour M1 2526/Classif/Demi_jour_datascience/farms_train.csv", header = TRUE)
test  <- read.csv("C:/Users/User/Desktop/Cour M1 2526/Classif/Demi_jour_datascience/farms_test.csv",  header = TRUE)

features <- c("R2","R7","R8","R17","R22","R32")
target   <- "DIFF"

stopifnot(all(features %in% names(train)))
stopifnot(all(features %in% names(test)))
stopifnot(target %in% names(train))


train <- train %>%
  mutate(DIFF_factor = factor(DIFF, levels = c(0,1)),
         DIFF_num    = as.numeric(as.character(DIFF)))


plot_box <- function(var){
  ggplot(train, aes(x = DIFF_factor, y = .data[[var]])) +
    geom_boxplot(aes(fill = DIFF_factor), alpha = 0.8) +
    labs(title = paste("DIFF vs", var), x = "DIFF (0=def, 1=saine)", y = var) +
    theme_minimal()
}


plot_box("R2");  plot_box("R7");  plot_box("R8")
plot_box("R17"); plot_box("R22"); plot_box("R32")


form_full <- as.formula(paste("DIFF_factor ~", paste(features, collapse = " + ")))
fit_full  <- glm(form_full, family = "binomial", data = train)
summary(fit_full)



pred_prob_train <- predict(fit_full, type = "response")

pred <- ROCR::prediction(pred_prob_train, train$DIFF_num)
perf <- ROCR::performance(pred, "tpr", "fpr")                     # TPR vs FPR
auc  <- ROCR::performance(pred, "auc")@y.values[[1]]

plot(perf, col = "blue", lwd = 2, main = sprintf("ROC — glm logistique (AUC = %.3f)", auc))
abline(a = 0, b = 1, col = "red", lty = 2)

auc

pred_prob_test <- predict(fit_full, newdata = test, type = "response")


alpha <- 1/2
pred_class_test <- ifelse(pred_prob_test >= alpha, 1, 0)


submission <- data.frame(
  ID   = seq_len(nrow(test)),
  DIFF = pred_class_test
)

write.csv(submission, "submission_logreg_labels.csv", row.names = FALSE)
