#################################################################
################ Data Science: Capstone! ########################
#################################################################

#### May 28th 2024 ####
#### By Sandra Lucía Rodríguez Ochoa ###


# Final course in the HarvardX Professional Certificate in Data Science.
# Create a movie recommendation system using the MovieLens dataset.
# R. Script

#---------------------------------------------------------
# 1. Introduction/overview
#---------------------------------------------------------

# This is a movie recommendation system based on the MovieLens dataset. 
# As per instructions just a small subset of data was extracted from original data base.

# The objective is train a machine learning algorithm using the inputs in one subset to predict movie ratings
# in the validation set.


# ----- 1.1 Download the MovieLens data and run the code provided to generate the datasets


# Create edx and final_holdout_test sets 
# Note: this process could take a couple of minutes

#if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# Load Required libraries
library(tidyverse)
library(caret)
library(dplyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)


# Download Ratings and movies files, convert to Data Frames assign column names and join in movielens

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")


# Prepare Final hold-out test set to be used for evaluating the RMSE of the final algorithm.

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)


#---------------------------------------------------------
# 2. Methods & analysis 
# section that explains the process and techniques used, including data cleaning, 
# data exploration and visualization, insights gained, and your modeling approach
#---------------------------------------------------------

# ---- 2.1 Review DataSet and get familiar with the data

class(edx)
dim(edx)
summary(edx)
range(edx$rating)


# ********* Create a column with year of movie release extracted from title
edx<-edx%>% mutate(year=as.integer(str_sub(title,-5,-2)))
range(edx$year)

#top 5 earliest and latest releases
edx%>%group_by(movieId)%>%summarize (year=first(year), title = first(title))%>%arrange(year)%>%head(10)
edx%>%group_by(movieId)%>%summarize (year=first(year), title = first(title))%>%arrange(year)%>%tail(10)

# ********* Create a column with year of rating extracted from timestamp
edx <- mutate(edx, year_rating = year(as_datetime(timestamp)))
range(edx$year_rating)



# ---- 2.2 Data Set Exploratory analysis

#Rating distribution
edx%>%ggplot(aes(rating))+ geom_histogram(bins = 10, fill = "darkslateblue", alpha = 0.7, col="white") +
  scale_y_continuous(labels = scales::comma_format())+ggtitle("Rating Distribution")


#*** mean rating by movie
movie_mean <- edx%>%group_by(movieId)%>%summarize(n=n(),avg_rating=mean(rating))%>%arrange(desc(avg_rating))
movie_mean%>%ggplot(aes(movieId,avg_rating,color=n))+geom_point()+ggtitle("Mean rating by movieId")

#*** Number of ratings per movie distribution
movie_mean%>%ggplot(aes(n)) + geom_histogram(bins = 100,fill = "darkslateblue", alpha = 0.7, col="white")+
  coord_trans(y = "sqrt") + ggtitle("Number of ratings per movie distribution")

#*** Number of ratings per userId distribution
edx%>%group_by(userId)%>%summarize(n=n(),avg_rating=mean(rating))%>%arrange(desc(n))%>%  
  ggplot(aes(n))+ geom_histogram(bins = 100, fill = "darkslateblue", alpha = 0.7, col="white")+
  coord_trans(y = "sqrt") + ggtitle("Number of ratings per user distribution")

#*** Number of reatings per year
edx%>%group_by(year)%>%summarize(n=n(),avg_rating=mean(rating))%>%arrange(desc(avg_rating))%>%
  ggplot(aes(year,n))+ geom_col( fill = "darkslateblue", alpha = 0.7, col="white") + ggtitle("Number of ratings per year")

#*** Box plot rating per year after 1980
edx %>% group_by(movieId) %>%
  summarize(n = n(), avg_rating=mean(rating), year = as.character(first(year)))%>%
  qplot(year, avg_rating, data = ., geom = "boxplot") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#*** Box plot Avg rating per genre per movie

movie_genre<-edx%>% filter(grepl("Drama|Comedy|Thriller|Romance", genres))%>%
  mutate(g=ifelse(grepl("Drama",genres),"Drama",ifelse(grepl("Romance",genres),"Romance",ifelse(grepl("Thriller",genres),"Thriller","Comedy"))))%>%
  group_by(movieId)%>% summarize(n = n(), avg_rating=mean(rating), genre = as.character(first(g)))


movie_genre%>%qplot(genre, avg_rating, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#*** Box plot count ratings per genre per movie
movie_genre%>%qplot(genre, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#scatter rating by movie by genre
movie_genre%>%ggplot(aes(movieId,avg_rating, color=genre))+geom_point()

#Rating years
edx%>%group_by(year_rating)%>% summarize(n = n(), avg_rating=mean(rating))%>%
  ggplot(aes(year_rating,n))+geom_col(fill = "darkslateblue", alpha = 0.7, col="white")+
  ggtitle("Number of ratings per year of rating")

edx%>%group_by(year_rating)%>% summarize(n = n(), avg_rating=mean(rating))%>%
  ggplot(aes(year_rating,avg_rating))+geom_col(fill = "darkslateblue", alpha = 0.7, col="white")+
  ggtitle("Average rating per year of rating")

#*** Box plot Avg rating per genre per movie

edx %>% group_by(movieId) %>%filter(genres %in% c("Drama", "Comedy", "Thriller", "Romance"))%>%
  summarize(n = n(), avg_rating=mean(rating), genre = as.character(first(genres)))%>%
  qplot(genre, avg_rating, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#**** Top 10 movies with greatest number of ratings
edx%>%group_by(movieId)%>%summarize(n=n(),title=as.character(first(title)),avg_rating=mean(rating))%>%
  arrange(desc(n))%>%head(10)%>%
  ggplot(aes(title,n))+geom_col(fill = "darkslateblue", alpha = 0.7, col="white")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#**** Top 10 genres with greatest number of ratings
edx%>%group_by(genres)%>%summarize(n=n(),genre=as.character(first(genres)),avg_rating=mean(rating))%>%
  arrange(desc(n))%>%head(10)%>%
  ggplot(aes(genre,n))+geom_col(fill = "darkslateblue", alpha = 0.7, col="white")+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))


#---------------------------------------------------------
# 4. Split data in Train and test sets
#---------------------------------------------------------

# ******* Divide edx in test 20% and train 80% sets

movietest_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
movie_test <- edx[movietest_index,]
movie_train <- edx[-movietest_index,]

movie_test <- movie_test %>% 
  semi_join(movie_train, by = "movieId") %>%
  semi_join(movie_train, by = "userId")

movie_test_c<-movie_test
movie_train_c<- movie_train



#---------------------------------------------------------
# 4. Loss Function RMSE Root Mean Squaered Error
#---------------------------------------------------------

#The Root mean squared error (RMSE) will be used as loss function. 
# RMSE is defined as follows: 

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#---------------------------------------------------------
# 5. Training Data and Modeling final algorithm
#---------------------------------------------------------
# The next step is to build models and compare them to each other.

#*** First Model, the same value for all movies accross all users. The mean is the value that minimize the error
#*** General raiting mean across all users and all movies
mu<-mean(movie_train$rating) 
#Predict with test set
pred_mu<-rep(mu,nrow(movie_test))
#Calculate the RMSE
rmse_mu<-RMSE(movie_test$rating,pred_mu)

#**** Adding Movie effect
#-----------------------------
movie_coef<-movie_train%>%group_by(movieId)%>%summarize(bi=mean(rating-mu))
movie_train<-left_join(movie_train,movie_coef,by="movieId")


movie_test<-left_join(movie_test,movie_coef,by="movieId")
#Predict with test set
pred_mu_bi<-pred_mu+movie_test$bi
#Calculate the RMSE
rmse_mu_bi<-RMSE(movie_test$rating,pred_mu_bi)


#**** Adding User effect
#*#-----------------------------
user_coef<-movie_train%>%group_by(userId)%>%summarize(bu=mean(rating-bi-mu))
movie_train<-left_join(movie_train,user_coef,by="userId")

movie_test<-left_join(movie_test,user_coef,by="userId")
#Predict with test set
pred_mu_bi_bu<-pred_mu+movie_test$bi+movie_test$bu
#Calculate the RMS
rmse_mu_bi_bu<-RMSE(movie_test$rating,pred_mu_bi_bu)

#Adding genre effect
#*#-----------------------------
genre_coef <- movie_train%>%group_by(genres)%>%summarize(bg=mean(rating-bu-bi-mu))
movie_train<-left_join(movie_train,genre_coef,by="genres")

movie_test<-left_join(movie_test,genre_coef,by="genres")
#Predict with test set
pred_mu_bi_bu_bg<-pred_mu+movie_test$bi+movie_test$bu+movie_test$bg
#Calculate the RMS
rmse_mu_bi_bu_bg<-RMSE(movie_test$rating,pred_mu_bi_bu_bg)

#Adding release year effect
#*#-----------------------------
year_coef <- movie_train%>%group_by(year)%>%summarize(by=mean(rating-bg-bu-bi-mu))
movie_train<-left_join(movie_train,year_coef,by="year")

movie_test<-left_join(movie_test,year_coef,by="year")
#Predict with test set
pred_mu_bi_bu_bg_by<-pred_mu+movie_test$bi+movie_test$bu+movie_test$bg+movie_test$by
#Calculate the RMS
rmse_mu_bi_bu_bg_by<-RMSE(movie_test$rating,pred_mu_bi_bu_bg_by)


#Adding raiting Year Effect january 1 1970

yrating_coef<-movie_train%>%group_by(year_rating)%>%summarize(byr=mean(rating-by-bg-bu-bi-mu))
movie_train<-left_join(movie_train,yrating_coef,by="year_rating")

movie_test<-left_join(movie_test,yrating_coef,by="year_rating")
#Predict with test set
pred_mu_bi_bu_bg_by_byr<-pred_mu+movie_test$bi+movie_test$bu+movie_test$bg+movie_test$by+movie_test$byr
#Calculate the RMS
rmse_mu_bi_bu_bg_by_byr<-RMSE(movie_test$rating,pred_mu_bi_bu_bg_by_byr)


#---------------------------------------------------------
# 5. Apply regularization to improve results
#---------------------------------------------------------

#Some movies were rated by very few users, in most cases just 1. 
#there's a lot of uncertainty in the estimation. Therefore, larger estimates of , negative or positive, are more likely.
#Large errors can increase our RMSE, so it would rather be conservative when unsure.
#Regularization permits us to penalize large estimates that are formed using small sample sizes.

#Regularization, lets test with a 2 to 6 lambda sequence
lambda<-seq(2,6,0.1)

rmses_lambda <- sapply(lambda, function(l){
  bi <- movie_train_c %>% 
    group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+l))
  
  bu <- movie_train_c %>% left_join(bi, by="movieId") %>%
    group_by(userId) %>% summarize(bu = sum(rating - bi - mu)/(n()+l))
  
  bg <- movie_train_c %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>%
    group_by(genres)  %>% summarize(bg = sum(rating - bu - bi - mu)/(n()+l))
  
  by <- movie_train_c %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>% left_join(bg, by="genres") %>%
    group_by(year) %>% summarize(by = sum(rating - bg - bu - bi - mu)/(n()+l))
  
  byr <- movie_train_c %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>% left_join(bg, by="genres") %>% left_join(by, by="year") %>%
    group_by(year_rating) %>% summarize(byr = sum(rating - by - bg - bu - bi - mu)/(n()+l))
  
  predict <- movie_test_c %>%
    left_join(bi, by="movieId") %>%
    left_join(bu, by="userId") %>%
    left_join(bg, by="genres") %>%
    left_join(by, by="year") %>%
    left_join(byr, by="year_rating")%>%
    mutate(pred =  mu+ bi + bu + bg + by + byr) %>% pull(pred)
  
  return(RMSE(predict, movie_test_c$rating))
})
rmses_lambda

#identify the lambda that minimize the error
plot(lambda, rmses_lambda)

rmse_min <- min(rmses_lambda)
best_lambda<-lambda[which.min(rmses_lambda)]
best_lambda


#---------------------------------------------------------
# Results
#---------------------------------------------------------
# Lets compare the different models

rmses<- matrix(c(rmse_mu,rmse_mu_bi,rmse_mu_bi_bu, rmse_mu_bi_bu_bg, rmse_mu_bi_bu_bg_by, rmse_mu_bi_bu_bg_by_byr,rmse_min), ncol=1, byrow=TRUE)
rownames(rmses)<-c("Mean","Movie Effect","Movie and User Effect", "Movie, User and Genre Effect","Movie, User, Genre and Release year Effect","Movie, User, Genre, Release year and Rating year Effect", "Regularization")
colnames(rmses)<- "RMSE"

as.table(rmses)

#Predict rating of Final Hold out set with final algorithm

l=best_lambda

bi <- edx %>% 
  group_by(movieId) %>% summarize(bi = sum(rating - mu)/(n()+l))

bu <- edx %>% left_join(bi, by="movieId") %>%
  group_by(userId) %>% summarize(bu = sum(rating - bi - mu)/(n()+l))

bg <- edx %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>%
  group_by(genres)  %>% summarize(bg = sum(rating - bu - bi - mu)/(n()+l))

by <- edx %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>% left_join(bg, by="genres") %>%
  group_by(year) %>% summarize(by = sum(rating - bg - bu - bi - mu)/(n()+l))

byr <- edx %>% left_join(bi, by="movieId") %>% left_join(bu, by="userId") %>% left_join(bg, by="genres") %>% left_join(by, by="year") %>%
  group_by(year_rating) %>% summarize(byr = sum(rating - by - bg - bu - bi - mu)/(n()+l))

# ********* Create a column with year of movie release extracted from title
final_holdout_test_<-final_holdout_test

final_holdout_test_<-final_holdout_test_%>% mutate(year=as.integer(str_sub(title,-5,-2)))
final_holdout_test_ <- mutate(final_holdout_test_, year_rating = year(as_datetime(timestamp)))

predict <- final_holdout_test_ %>%
  left_join(bi, by="movieId")%>%
  left_join(bu, by="userId") %>%
  left_join(bg, by="genres") %>%
  left_join(by, by="year") %>%
  left_join(byr, by="year_rating")%>%
  mutate(pred =  mu+ bi + bu + bg + by + byr) %>% pull(pred)

rmse_final_holdout<-(RMSE(predict, final_holdout_test_$rating))
rmse_final_holdout

#---------------------------------------------------------
#Conclusion section that gives a brief summary of the report, its limitations and future work
#---------------------------------------------------------
