#install.packages("readr")

#Importation des packages
library(corrplot)
library(readr)
library(mlbench)
library(class)  # pour knn
library(caret)  # pour diviser les données
library(ggplot2)

getwd()

train <- read_csv("farms_train.csv")
test <- read_csv("farms_test.csv")

#Vérification des valeurs manquantes
sum(is.na(train))
sum(is.na(test))

#Description des tables (jeu entraînement) -> 7 variables pour 401 valeurs
dim(train)
nrow(train)

# Résumé statistique par colonne (jeu entraînement)
summary(train)

#Description des tables (jeu test) -> 6 variables pour 119 valeurs, on a un jeu plus petit
dim(test)
nrow(test)

# Résumé statistique par colonne (jeu test)
summary(test)

set.seed(10) #Permet de garder le même aléatoire
Index <- createDataPartition(train$DIFF, p = 0.7, list = FALSE)
data <- train[Index,]

x <- data[, -1]
y <- data$DIFF

knn.pred <- knn(x,x, cl = y, k = 5)

mean(y!=knn.pred) #On obtient un taux d'erreur égale à 0.181 soit 18%
1 - mean(y!=knn.pred) #On obtient une précision de 0.818 soit 81%

#On réalise la même chose avec l'ensemble des données, où le jeu d'entraînement est le jeu entier sauf la variable diff
set.seed(10)  # Pour rendre le partitionnement reproductible


Index <- createDataPartition(train$DIFF, p = 0.7, list = FALSE)
trainData <- train[Index, ]
testData  <- train[-Index, ]

x_train <- trainData[, -1]   # toutes les colonnes sauf DIFF
y_train <- trainData$DIFF

x_test  <- testData[, -1]
y_test  <- testData$DIFF

knn.predEnsemble <- knn(train = x_train, test = x_test, cl = y_train, k = 5)

Erreur <- mean(y_test!=knn.predEnsemble)
1 - Erreur

plot(Erreur, type="b", col="blue", cex=1, pch=20,
     xlab="K, nombre de voisins", ylab="Taux d'erreur (%)", 
     main="Taux d'erreur vs Nombre de voisins")


# Lignes horizontale identifiant les valeurs de K pour lesquelles on a le plus petit taux d'erreur
abline(v=which(Erreur==min(Erreur)), col="green", lwd=1.5)

#Ligne verticale indiquant quelle est la valeur maximale du taux d'erreur observé
abline(h=max(Erreur), col="red", lty=2)

#Ligne verticale indiquant quelle est la valeur minimale du taux d'erreur observé
abline(h=min(Erreur), col="red", lty=2)


k_values = 1:100
tauxerreur <- numeric(length(k_values))

for(i in 1:length(k_values)){
  #réalisation de la prédiction
  knn.pred<-knn(x_train,x_test,y_train,k=i)
  #Précision du modèle
  tauxerreur[i]=mean(y_test!=knn.pred)*100
}

plot(tauxerreur, type="b", col="blue", cex=1, pch=20,
     xlab="K, nombre de voisins", ylab="Taux d'erreur (%)", 
     main="Taux d'erreur vs Nombre de voisins")


# Lignes horizontale identifiant les valeurs de K pour lesquelles on a le plus petit taux d'erreur
abline(v=which(tauxerreur==min(tauxerreur)), col="green", lwd=1.5)

#Ligne verticale indiquant quelle est la valeur maximale du taux d'erreur observé
abline(h=max(tauxerreur), col="red", lty=2)

#Ligne verticale indiquant quelle est la valeur minimale du taux d'erreur observé
abline(h=min(tauxerreur), col="red", lty=2)

#Avec ce graphique obtenu on se rends compte sur le taux d'erreur est très élevé quand k est égale à 2 (C'est là ou il fait le plus d'erreur)
#(avec plus de 35%), le problème peut venir de bruit du jeu de donnée c'est ce qu'on appelle un problème de sur-apprentissage

#On remarque qu'ensuite à partir de k=3 une chute brutale de l'erreur,qui montre que par la suite le modèle fait de moins en moins d'erreur

#On peut voir également que les k optimal, ou il y a le moins d'erreur se trouve pour k=22, 24, 32, 34 et 86
#Grâce à ce graphique on peut observer le comportement du modèle KNN.



#Vue précédemment, je prends k=22 car il était l'un des k optimales

k_optimal <- 22

knn.pred.final <- knn(train = x_train, test = x_test, cl = y_train, k = k_optimal)

#Création du dataframe pour l'exporter sous kaggle
id <- 0:(length(knn.pred.final) - 1)
result_df <- data.frame(Id = id, Diff = knn.pred.final)
write.csv(result_df, "predictions_knn.csv", row.names = FALSE)


Erreurfinal <- mean(y_test!=knn.pred.final) 
1 - Erreurfinal
#Précision de 0.783 et 78%

final <- read_csv("predictions_knn.csv")


