# Libraries

load.libraries <- c('data.table', 'testthat', 'gridExtra', 'corrplot', 'GGally', 'ggplot2', 'e1071', 'dplyr')
sapply(load.libraries, require, character = TRUE)

# Funções auxiliares
plot_Missing <- function(data_in, title = NULL){
  temp_df <- as.data.frame(ifelse(is.na(data_in), 0, 1))
  temp_df <- temp_df[,order(colSums(temp_df))]
  data_temp <- expand.grid(list(x = 1:nrow(temp_df), y = colnames(temp_df)))
  data_temp$m <- as.vector(as.matrix(temp_df))
  data_temp <- data.frame(x = unlist(data_temp$x), y = unlist(data_temp$y), m = unlist(data_temp$m))
  ggplot(data_temp) + geom_tile(aes(x=x, y=y, fill=factor(m))) + scale_fill_manual(values=c("white", "black"), name="Missing\n(0=Yes, 1=No)") + theme_light() + ylab("") + xlab("") + ggtitle(title)
}

convert_data <- function(data){

  a = which(sapply(data, is.numeric) == FALSE)
  a = unname((a))
  for (i in a){
    data[, i] = as.numeric(data[, i])
  }
  return (data)
}


# Set the working directory.
setwd("C:\\Users\\marco\\Documents\\Git\\HousesPrices\\data")
# source('C:\\Users\\marco\\Documents\\Git\\HousesPrices\\analysing.R')

trainset = read.csv("train.csv", sep=",")
# setDT(trainset)
# Separação do trainset em X e Y.
X = trainset[,1:ncol(trainset) - 1]
Y = trainset[,ncol(trainset)]

# Variaveis numéricas e alfa-numéricas
cat_var <- names(trainset)[which(sapply(trainset, is.character))]
cat_car <- c(cat_var, 'BedroomAbvGr', 'HalfBath', ' KitchenAbvGr','BsmtFullBath', 'BsmtHalfBath', 'MSSubClass')
numeric_var <- names(trainset)[which(sapply(trainset, is.numeric))]

# Dimensão do trainset.
# dim(trainset)

# Visão dos dados uma coluna por linha .
# str(train)

# Visão dos primeiros elementos.
# head(train)

# Contagem de quantos elementos são nulos em cada coluna.
# colSums(sapply(trainset, is.na))

# Grafico para a visualização das variaveis nulas.
# plot_Missing(trainset[,colSums(is.na(trainset)) > 0])


######### Algumas informações sobre os dados ###########
# Casas reformadas.
n_remodeled = sum(trainset[,'YearRemodAdd'] != trainset[,'YearBuilt'])
#cat('Percentage of houses remodeled:', n_remodeled/ dim(trainset)[1])
# train %>% select(YearBuilt, YearRemodAdd) %>%    mutate(Remodeled = as.integer(YearBuilt != YearRemodAdd)) %>% ggplot(aes(x= factor(x = Remodeled, labels = c( 'No','Yes')))) + geom_bar() + xlab('Remodeled') + theme_light()

# Dados faltando na base
missing_data = sum(is.na(trainset)) / (nrow(trainset) *ncol(trainset))

## resumo dos dados numéricos
# summary(trainset[,.SD, .SDcols =numeric_var])

## Removendo dados categoricos
train = convert_data(trainset)
