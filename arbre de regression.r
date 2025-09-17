
# Chargement des librairies

packages <- c("tidyverse","rpart","rpart.plot","caret","pROC")
to_install <- setdiff(packages, installed.packages()[,"Package"])
if(length(to_install)) install.packages(to_install)
lapply(packages, library, character.only = TRUE)

set.seed(42)  


# etape1 Préparer les données
# (on suppose que farms_train et farms_test sont déjà en mémoire)

train_data <- farms_train
test_data  <- farms_test

# Corriger les niveaux de la variable cible
# 1 = sain (positif), 0 = def (négatif)
train_data$DIFF <- factor(train_data$DIFF,
                          levels = c(1, 0),
                          labels = c("sain", "def"))

levels(train_data$DIFF)   # Vérification -> doit afficher: "sain" "def"


# Etape 2 — Vérifier NA + imputation simple
dim(train_data); dim(test_data)
colSums(is.na(train_data)); colSums(is.na(test_data))

impute_med <- function(df){
  num <- sapply(df, is.numeric)
  for(v in names(df)[num]){
    df[[v]][is.na(df[[v]])] <- median(df[[v]], na.rm = TRUE)
  }
  df
}
train_data <- impute_med(train_data)
test_data  <- impute_med(test_data)


# etap 3 — Définir la validation croisée (AUC ROC)

ctrl <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)


# etape 4 — Tuning de l’arbre (cp × profondeur)

cp_grid <- data.frame(cp = seq(0.000, 0.05, by = 0.005))
depths  <- c(3,4,5,6,8)

fits <- list(); all_results <- list()

for (d in depths) {
  fit <- caret::train(
    DIFF ~ ., data = train_data,
    method = "rpart",
    metric = "ROC",
    trControl = ctrl,
    tuneGrid  = cp_grid,
    control   = rpart.control(maxdepth = d, minsplit = 10, minbucket = 5)
  )
  fits[[paste0("depth_", d)]] <- fit
  all_results[[length(all_results)+1]] <- dplyr::mutate(fit$results, maxdepth = d)
}

res <- dplyr::bind_rows(all_results)


# Etap 5 — Choisir le meilleur modèle

res_clean <- res[!is.na(res$ROC), ]
ix <- which.max(res_clean$ROC)
best <- res_clean[ix, , drop = FALSE]

best_depth <- as.numeric(best$maxdepth[1])
best_cp    <- as.numeric(best$cp[1])
best_auc   <- as.numeric(best$ROC[1])
best_fit   <- fits[[paste0("depth_", best_depth)]]

cat("Meilleur arbre — depth:", best_depth,
    "| cp:", best_cp, "| AUC(CV):", round(best_auc,4), "\n")


# Etp 6 — Interprétation (importance + visualisation)

print(varImp(best_fit))
rpart.plot(best_fit$finalModel, type = 2, extra = 104, under = TRUE, faclen = 0)


# etp 7 — Prédire les probabilités sur le test

proba_test <- predict(best_fit, newdata = test_data, type = "prob")[,"sain"]
stopifnot(length(proba_test) == nrow(test_data))
summary(proba_test)


# Etape 8 — Construire le fichier de soumission

if (exists("exsub") && "ID" %in% names(exsub)) {
  submission <- exsub %>% dplyr::mutate(DIFF = proba_test)
} else {
  submission <- data.frame(ID = seq_len(nrow(test_data)), DIFF = proba_test)
}

write.csv(submission, "soumission_decision_tree.csv", row.names = FALSE)
cat(" Fichier créé : soumission_decision_tree.csv\n")

# Prédictions sur le TRAIN avec seuil alpha  0.5

# Prédictions sur le fichier test (seuil alpha = 0.5)


# 1) Probabilités de la classe "sain"
proba_test <- predict(best_fit, newdata = test_data, type = "prob")[,"sain"]

# 2) Conversion en classes 0/1 avec alpha = 0.5
pred_test_01 <- ifelse(proba_test >= 0.5, 1, 0)

# 3) Construire la table de sortie
#   
if ("ID" %in% names(test_data)) {
  submission <- data.frame(ID = test_data$ID, DIFF = pred_test_01)
} else {
  submission <- data.frame(ID = seq_len(nrow(test_data)), DIFF = pred_test_01)
}

# 4) Sauvegarde en CSV
write.csv(submission, "predictions_test.csv", row.names = FALSE)

# 5) Vérifications rapides
nrow(submission)        
head(submission)
table(submission$DIFF)  

getwd()