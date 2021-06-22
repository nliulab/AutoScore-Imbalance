#Load necessary packages
library(pROC)
library(reticulate)
library(rpart)
library(DMwR)
library(mltools)
library(AutoScore)

#When meeting "Error in plot.new() : figure margins too large", users can try clear all plots
# AutoScore-Imbalance pipeline function --------------------------------------------------------
#auto calculator for different imbalanced adjustment measures
Training_data_optimizing<-function(data_train, data_validation,
                          method, var_num, 
                          random_seed_out = 1234, steps = 1, gan_positive_results = 1){
  set.seed(random_seed_out)
  a<-length(which(data_train$label==1))
  b<-nrow(data_train)
  generated_numbers<-((50-ceiling(a/b*100))/steps+1)
  generated_dataset<-list()
  generated_dataset[[1]]<-data_train
  generated_ratio<-seq(from = ceiling((a/b)*100), to = 50, by = steps)/100
  all_ratio<-c(round((a/b),4), generated_ratio)
  #AUPRC_values<-c(1:(length(generated_ratio)+1))
  AUROC_values<-rep(0,(length(generated_ratio)+1))
  if (method == "smote"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-smote_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("smote dataset", i, "has been generated"))
    }}
  if (method == "upsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-upsampling_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("Upsampling dataset", i, "has been generated"))
    }}
  if (method == "downsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-downsampling_generation(data_train,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("Downsampling dataset", i, "has been generated"))
    }}
  if (method == "up_and_down_sampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-up_and_down_generation(data_train,generated_ratio[i],random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("up_and_down_sampling dataset", i, "has been generated"))
    }}
  if (method == "smote_and_down_sampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-smote_and_down_generation(data_train,generated_ratio[i],random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("smote_and_down_sampling dataset", i, "has been generated"))
    }}
  if (method == "gan"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-gan_generation(data_train,gan_positive_results,generated_ratio[i],random_seed = random_seed_out)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated raio is",rr))
      print(paste("gan dataset", i, "has been generated"))
    }}
  if (method == "gan_and_downsampling"){
    for (i in 1:length(generated_ratio)) {
      generated_dataset[[i+1]]<-gan_and_down_generation(dataset_out = data_train, ratio_out = generated_ratio[i], 
                              random_seed_out,gan_positive_results)
      rr<-length(which(generated_dataset[[i+1]]$label==1))/nrow(generated_dataset[[i+1]])
      print(paste("real generated ratio is",rr))
      print(paste("gan_and_downsampling dataset", i, "has been generated"))
    }}
  for (j in 1:(length(generated_ratio)+1)) {
    rr<-length(which(generated_dataset[[j]]$label==1))/nrow(generated_dataset[[j]])
    print(paste("current raio is",rr))
    Ranking<-AutoScore_rank(generated_dataset[[j]])
    #AUC<-AutoScore_parsimony(generated_dataset[[j]],data_validation,Ranking)
    FinalVariable<-names(Ranking)[1:var_num]
    CutVec<-AutoScore_weighting(generated_dataset[[j]],data_validation,FinalVariable)
    ScoringTable<-AutoScore_fine_tuning(generated_dataset[[j]],data_validation,FinalVariable,CutVec)
    test_result<-auc_calculating(data_validation,FinalVariable,CutVec,ScoringTable)
    #AUPRC_values[j]<-test_result[[7]]$auc.integral
    AUROC_values[j]<-test_result
    print(paste("Dataset", j, "Autoscore has been calculated"))
  }
  optimal_data<-generated_dataset[[which(AUROC_values==max(AUROC_values[-1]))]]
  return_values<-list(AUROC = AUROC_values, optimal_dataset = optimal_data)
  return(return_values)
}

Sample_weights_optimizing<-function(data_train,data_validation,predictor,predictor_num,steps = 1,random_seed_out = 1234){
  set.seed(random_seed_out)
  weight_max<-length(which(data_train$label==0))/length(which(data_train$label==1))
  weight_list<-c(seq(from = 1, to = ceiling(weight_max), by = steps))
  weight_list<-round(weight_list,0)
  sample_weight<-c(rep(1,nrow(data_train)))
  auroc_list<-c(rep(1,length(weight_list)))
  for (i in 1:length(weight_list)) {
    weight_tmp<-weight_list[i]
    sample_weight_tmp<-sample_weight
    sample_weight_tmp[which(data_train$label==1)]<-weight_tmp
    #auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
    final_variable<-names(predictor)[1:predictor_num]
    auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
    myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
    test_result<-auc_calculating(data_validation,final_variable,auto_weight,myvec)
    auroc_list[i]<-test_result
    print(paste("Adjusted weight", i,"in",length(weight_list), "has been calculated"))
  }
  weight_final_tmp<-c()
  for (i in 1:length(weight_list)) {
    #if (specificity_list[i]>=specificity_list[1] & sensitivity_list[i]>=sensitivity_list[1]){weight_final_tmp<-c(weight_final_tmp,i)}
    if (auroc_list[i]>=auroc_list[1]){weight_final_tmp<-c(weight_final_tmp,i)}
  }
  weight_final<-weight_list[which(auroc_list==max(auroc_list[weight_final_tmp]))]
  sample_weight_tmp<-sample_weight
  sample_weight_tmp[which(data_train$label==1)]<-weight_final
  #auc_parsimony<-AutoScore_parsimony_weight(data_train,data_validation,predictor,weight = sample_weight_tmp)
  final_variable<-names(predictor)[1:predictor_num]
  auto_weight<-AutoScore_weighting_weight(data_train,data_validation,final_variable,weight = sample_weight_tmp)
  myvec<-AutoScore_fine_tuning_weight(data_train,data_validation,final_variable,auto_weight,weight = sample_weight_tmp)
  results<-list(final_variable,auto_weight,myvec,sample_weight_tmp)
  return(results)
}

# Data generation function --------------------------------------------------------
#Data generation by SMOTE
smote_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  dataset$label<-as.factor(dataset$label)#important due to smote requirement
  ratio_smote<-c(0,0)
  result<-c(0,0)
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-(b/(1-ratio)-b)%/%a
  int<-(c-1)*100
  residual<-round(((b/(1-ratio)-b)/a-c)*100)
  result[1]<-int
  result[2]<-residual
  ratio_smote<-result
  if (ratio_smote[1]<=0 & a*ratio_smote[2]/100<2){return(dataset)}
  if (ratio_smote[1]<=0 & a*ratio_smote[2]/100>=2){
    set.seed(random_seed)
    data_generation<-SMOTE(label~ ., dataset, perc.over = ratio_smote[2], perc.under = 100)
    data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
    data_all<-data_all[complete.cases(data_all),]
    return(data_all)
  }
  if (ratio_smote[1]>0 & a*ratio_smote[2]/100<2){
    set.seed(random_seed)
    data_generation<-SMOTE(label~ ., dataset, perc.over = ratio_smote[1], perc.under = 100)
    data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
    data_all<-data_all[complete.cases(data_all),]
    return(data_all)
  }
  d1<-SMOTE(label~., dataset, perc.over = ratio_smote[1], perc.under = 100)
  d2<-SMOTE(label~., dataset, perc.over = ratio_smote[2], perc.under = 100)
  d1<-d1[complete.cases(d1),]
  d2<-d2[complete.cases(d2),]
  {if (nrow(d1)==0) {return(d2)}}
  {if (nrow(d2)==0) {return(d1)}}
  data_generation<-rbind(d1,d2)
  data_all<-rbind(data_generation[which(data_generation$label==1),], dataset[which(dataset$label==0),])
  return(data_all)
}

#AutoScore-Imbalance data generation by upsampling
upsampling_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  positive_data<-dataset[which(dataset$label==1),]
  negative_data<-dataset[which(dataset$label==0),]
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-round(b/(1-ratio)-b)
  d<-c%%a
  e<-c%/%a
  data_all<-rbind(positive_data[as.vector(sample(c(1:nrow(positive_data)),size = d,replace = F)),],negative_data)
  for (i in 1:e) {
    data_all<-rbind(data_all,positive_data)
  }
  return(data_all)
}

#Data generation by downsampling
downsampling_generation<-function(dataset, ratio, random_seed = 1234){
  set.seed(random_seed)
  positive_data<-dataset[which(dataset$label==1),]
  negative_data<-dataset[which(dataset$label==0),]
  a<-length(which(dataset$label==1))
  c<-round(a/ratio-a)
  data_all<-rbind(positive_data,negative_data[as.vector(sample(c(1:nrow(negative_data)),size = c,replace = F)),])
  return(data_all)
}

up_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_up<-upsampling_generation(dataset = dataset_out,ratio = (ratio_out+raw_ratio)/2, random_seed = random_seed_out)
  dataset_down<-downsampling_generation(dataset = dataset_up,ratio = ratio_out, random_seed = random_seed_out)
  return(dataset_down)
}

smote_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_smote<-smote_generation(dataset = dataset_out,ratio = ((ratio_out+raw_ratio)/2), random_seed = random_seed_out)
  if (length(which(dataset_smote$label==1))/nrow(dataset_smote)>=ratio_out) 
  {return(dataset_smote)}
  else {
    dataset_down<-downsampling_generation(dataset = dataset_smote,ratio = ratio_out, random_seed = random_seed_out)
    return(dataset_down)}
}

gan_and_down_generation<-function(dataset_out, ratio_out, random_seed_out = 1234,gan_positive_results){
  set.seed(random_seed_out)
  raw_ratio<-length(which(dataset_out$label==1))/nrow(dataset_out)
  dataset_gan<-gan_generation(dataset = dataset_out,gan_positive_result = gan_positive_results,ratio = ((ratio_out+raw_ratio)/2), random_seed = random_seed_out)
  if (length(which(dataset_gan$label==1))/nrow(dataset_gan)>=ratio_out) 
  {return(dataset_gan)}
  else {
    dataset_down<-downsampling_generation(dataset = dataset_gan,ratio = ratio_out, random_seed = random_seed_out)
    return(dataset_down)}
}

gan_preparation<-function(python_route){
  #set python environment
  use_condaenv(python_route)
  if (py_module_available("pandas")==T) {
    print("Python environment is available")
    print("Module pandas is available")} 
  else {print("error: Module pandas is not available")}
  if (py_module_available("ctgan")==T) {print("MOdule gan is available")} else {print("error: MOdule gan is not available")}
}

gan_positive_generation<-function(dataset, epoch, sample_num){
  ctgan<-import("ctgan")
  CTGANSynthesizer<-ctgan$CTGANSynthesizer
  discrete_column = "label"
  gan_fun = CTGANSynthesizer(epochs=epoch)
  gan_fun$fit(data_train, discrete_column)
  # Synthetic copy
  generated_data = gan_fun$sample(sample_num)
  print("Mutiple warning could be generated by gan in this step")
  positive_generated_sample<-generated_data[which(generated_data$label==1),]
  return(positive_generated_sample)
}

gan_generation<-function(dataset, gan_positive_result, ratio, random_seed = 1234){
  set.seed(random_seed)
  dataset_negative<-dataset[which(data_train$label==0),]
  a<-length(which(dataset$label==1))
  b<-length(which(dataset$label==0))
  c<-round(b/(1-ratio)-b-a)
  data_positive<-gan_positive_result[as.vector(sample(c(1:nrow(gan_positive_result)),
                                                      size = c,replace = T)),]
  data_all<-rbind(data_positive, dataset)
  return(data_all)
}

# AutoScore-Imbalance support function ------------------------------------------------

auc_calculating <- function(test_set, final_variables, cut_vec, scoring_table) {
  # prepare testset: categorization and "auto_test"
  test_set_1 <- test_set[, c(final_variables, "label")]
  test_set_2 <- transform_df_fixed(test_set_1, cut_vec = cut_vec)
  test_set_3 <- auto_test(test_set_2, scoring_table)
  test_set_3$total_score <- rowSums(subset(test_set_3, select = -label))
  test_set_3$total_score[which(is.na(test_set_3$total_score))]<-0
  y_test <- test_set_3$label
  model_roc <- roc(y_test, test_set_3$total_score, quiet = T)
  return_values<-round(auc(model_roc),3)
  return(return_values)
}

AutoScore_weighting_weight <- function(train_set, validation_set, final_variables, weight, max_score = 100,split = "quantile", quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1)) {
  # prepare train_set and VadalitionSet
  cat("****Included Variables: \n")
  print(data.frame(variable_name = final_variables))
  train_set_1 <- train_set[, c(final_variables, "label")]
  validation_set_1 <- validation_set[, c(final_variables, "label")]
  
  # AutoScore Module 2 : cut numeric and transfer categories and generate "cut_vec"
  df_transformed <- transform_df(train_set_1, validation_set_1, split = split, quantiles = quantiles, Print_categories = TRUE)
  train_set_2 <- df_transformed[[1]]
  validation_set_2 <- df_transformed[[2]]
  cut_vec_tmp <- df_transformed[[3]]
  cut_vec <- cut_vec_tmp
  for (i in 1:length(cut_vec)) cut_vec[[i]] <- cut_vec[[i]][2:(length(cut_vec[[i]]) - 1)]
  return(cut_vec)
}

AutoScore_fine_tuning_weight <- function(train_set, validation_set, final_variables, weight, cut_vec, max_score = 100) {
  # Prepare train_set and VadalitionSet
  train_set_1 <- train_set[, c(final_variables, "label")]
  validation_set_1 <- validation_set[, c(final_variables, "label")]
  
  # AutoScore Module 2 : cut numeric and transfer categories (based on fix "cut_vec" vector)
  train_set_2 <- transform_df_fixed(train_set_1, cut_vec = cut_vec)
  validation_set_2 <- transform_df_fixed(validation_set_1, cut_vec = cut_vec)
  
  # AutoScore Module 3 : Score weighting
  score_table<-compute_score_table_weight(train_set_2,validation_set_2,max_score,final_variables,weight)
  return(score_table)
}

transform_df <- 
  function(df,
           df_new,
           quantiles = c(0, 0.05, 0.2, 0.8, 0.95, 1), #by default
           Print_categories = FALSE, # >>> (Yilin) variable name starts with capital letter
           max_cluster = 5,
           split = "quantile") {
    
    # Generate cut_vec for downstream usage
    cut_vec <- list()
    
    for (i in 1:(length(df) - 1)) {
      # for factor variable
      if (class(df[, i]) == "factor") {
        if (length(levels(df[, i])) < 10)
          #(next)() else stop("Error!! The number of categories should be less than 10")
          (next)()
        else
          print(cat("Warning!! The number of categories should be less than 10"), # >>> (Yilin) Use warning function instead
                names(df)[i])
      }
      
      # for continuous variable: variable transformation
      # select discretization method, default mode = 1
      
      ## mode 1 - quantiles
      if (split == "quantile") {
        # options(scipen = 20)
        #print("in quantile")
        cut_off_tmp <- quantile(df[, i], quantiles = quantiles)
        cut_off_tmp <- unique(cut_off_tmp)
        cut_off <- signif(cut_off_tmp, 3)  # remain 3 digits
        #print(cut_off)
        
        ## mode 2 k-means clustering 
      } else if (split == "k_means") {
        #print("using k-means")
        clusters <- kmeans(df[, i], max_cluster)
        cut_off_tmp <- c()
        for (j in unique(clusters$cluster)) {
          #print(min(df[,i][clusters$cluster==j]))
          #print(length(df[,i][clusters$cluster==j]))
          cut_off_tmp <- append(cut_off_tmp, min(df[, i][clusters$cluster == j]))
          #print(cut_off_tmp)
        }
        cut_off_tmp <- append(cut_off_tmp, max(df[, i]))
        cut_off_tmp <- sort(cut_off_tmp)
        #print(names(df)[i])
        #assert (length(cut_off_tmp) == 6)
        cut_off_tmp <- unique(cut_off_tmp)
        cut_off <- signif(cut_off_tmp, 3)
        cut_off <- unique(cut_off)
        #print (cut_off)
        
        ## mode 3 decision_tree-rpart
      } else if (split == "decision_tree") {
        ## assign weights to address the umbalanced dataset
        # hardcode, might need to change
        w_positive <-
          nrow(df) / (length(unique(df$label)) * sum(df$label == 1))
        w_negative <-
          nrow(df) / (length(unique(df$label)) * sum(df$label == 0))
        #print(w_positive)
        #print(w_negative)
        w <-
          (as.numeric(df$label) - 1) * w_positive + (2 - as.numeric(df$label)) * w_negative
        
        m <- rpart(df$label ~ df[, i], method = 'class', weights = w)
        cut_off_tmp <- unname(m$splits[, max_cluster-1])
        cut_off_tmp <- head(cut_off_tmp, max_cluster-1)
        cut_off_tmp <- append(cut_off_tmp, max(df[, i]))
        cut_off_tmp <- append(cut_off_tmp, min(df[, i]))
        cut_off_tmp <- sort(cut_off_tmp)
        #print(names(df)[i])
        
        #assert (length(cut_off_tmp) == 6)
        cut_off_tmp <- unique(cut_off_tmp)
        #print(cut_off_tmp)
        cut_off <- signif(cut_off_tmp, 3)
        cut_off <- unique(cut_off)
        #print (cut_off)
        
      } else { # >>> (Yilin) Be consistent in how you write warning and error mesages, e.g., if you begin with "ERROR" or "WARNING" then do so for all
        stop('Error, please specify cut_off_tmp correct method for splitting:  quantile, k_means or decision_tree. input invalid!') 
      }
      
      
      
      # Generate cut_vec for downstream usage if (Print_categories == TRUE)
      if (Print_categories == TRUE)
      {
        #print(names(df)[i])
        # print(cut_off)
        l <- list(cut_off)
        #print("*****************************l***************************")
        #print(l)
        names(l)[1] <- names(df)[i]
        cut_vec <- append(cut_vec, l)
        #print("****************************cut_vec*************************")
        #print(cut_vec)
      }  #update
      
      # further processing for cut_off 
      if (length(cut_off) <= 2) {
        df[, i] <- as.factor(df[, i])
        df_new[, i] <- as.factor(df_new[, i])
      } else {
        #avoid produce NaN value at cut due to round down
        cut_off <- c(cut_off[cut_off < max(cut_off)], max(cut_off)*1.2)
        cut_off <- c(cut_off[cut_off > min(cut_off)], min(cut_off)*0.8)
        cut_off <- sort(cut_off)
        df[, i] <-
          cut(
            df[, i],
            breaks = cut_off,
            right = F,
            include.lowest = T,
            dig.lab = 3
          )
        
        ## delete min and max for the Interval after discretion: train_set
        # xmin<-unlist(strsplit(levels(df[,i])[1],','))[1] xmax<-unlist(strsplit(levels(df[,i])[length(levels(df[,i]))],','))[2]
        df[, i]<-delete_min_max(df[, i])
        
        
        ## make conresponding cutvec for validation_set: cut_off_newdata
        cut_off_newdata <- cut_off
        cut_off_newdata[1] <- min(cut_off_newdata[1], floor(min(df_new[, i])))
        cut_off_newdata[length(cut_off_newdata)] <- max(cut_off_newdata[length(cut_off_newdata)], ceiling(max(df_new[, i])))
        cut_off_newdata_1 <- signif(cut_off_newdata, 3)
        cut_off_newdata_1 <- unique(cut_off_newdata_1)  ###revised update##
        #print(cut_off_newdata_1)
        
        ## Cut make conresponding cutvec for validation_set: cut_off_newdata
        df_new[, i] <-
          cut(
            df_new[, i],
            breaks = cut_off_newdata_1,
            right = F,
            include.lowest = T,
            dig.lab = 3
          )
        # xmin<-as.character(min(cut_off_newdata_1)) xmax<-as.character(max(cut_off_newdata_1))
        
        ## delete min and max for the Interval after discretion: validation_set
        df_new[, i]<-delete_min_max(df_new[, i])
        
        
      }
      # print(summary(df[,i]))update print(summary(df_new[,i]))update
    }
    
    if (Print_categories == TRUE)
      return(list(df, df_new, cut_vec)) # >>> (Yilin) Name each element
    else
      return(list(df, df_new))
  }

compute_score_table_weight<-function(train_set_2,validation_set_2,max_score,variable_list,weight){
  #AutoScore Module 3 : Score weighting
  # First-step logistic regression
  model <- glm(label ~ ., family = binomial(link = "logit"), data = train_set_2,weights = weight)
  y_validation <- validation_set_2$label
  coef_vec <- coef(model)
  if (length(which(is.na(coef_vec)))>0) {warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
    coef_vec[which(is.na(coef_vec))]<-1}
  train_set_2 <- change_reference(train_set_2, coef_vec)
  
  # Second-step logistic regression
  model <- glm(label ~ ., family = binomial(link = "logit"), data = train_set_2,weights = weight)
  coef_vec <- coef(model)
  if (length(which(is.na(coef_vec)))>0) {warning(" WARNING: GLM output contains NULL, Replace NULL with 1")
    coef_vec[which(is.na(coef_vec))]<-1}
  
  # rounding for final scoring table "score_table"
  coef_vec_tmp <- round(coef_vec/min(coef_vec[-1]))
  score_table <- add_baseline(train_set_2, coef_vec_tmp)
  
  # normalization according to "max_score" and regenerate score_table
  total_max <- max_score
  total <- 0
  for (i in 1:length(variable_list)) total <- total + max(score_table[grepl(variable_list[i], names(score_table))])
  score_table <- round(score_table/(total/total_max))
  return(score_table)
}

change_reference <- function(df, coef_vec) {
  # delete label first
  df_tmp <- subset(df, select = -label)
  
  # one loops to go through all variable
  for (i in (1:length(df_tmp))) {
    char_tmp <- paste("^", names(df_tmp)[i], sep = "")
    coef_tmp <- coef_vec[grepl(char_tmp, names(coef_vec))]
    coef_tmp<- coef_tmp[!is.na(coef_tmp)]
    
    # if min(coef_tmp)<0, the current lowest one will be used for reference
    if (min(coef_tmp) < 0) {
      ref <- gsub(names(df_tmp)[i], "", names(coef_tmp)[which.min(coef_tmp)])
      df_tmp[, i] <- relevel(df_tmp[, i], ref = ref)
    }
  }
  # add lable again
  df_tmp$label <- df$label#df_tmp
  return(df_tmp)
}

add_baseline <- function(df, coef_vec) { # Proposed new version
  df <- subset(df, select = -label)
  coef_names_all <- unlist(lapply(names(df), function(var_name) {
    paste0(var_name, levels(df[, var_name]))
  }))
  coef_vec_all <- numeric(length(coef_names_all))
  names(coef_vec_all) <- coef_names_all
  # Remove items in coef_vec that are not meant to be in coef_vec_all 
  # (i.e., the intercept)
  coef_vec_core <- coef_vec[which(names(coef_vec) %in% names(coef_vec_all))]
  i_coef <- match(x = names(coef_vec_core), table = names(coef_vec_all))
  coef_vec_all[i_coef] <- coef_vec_core
  coef_vec_all
}

transform_df_fixed <- function(df, cut_vec = cut_vec) {
  j <- 1
  
  # for loop going through all variables
  for (i in 1:(length(df) - 1)) {
    
    if (class(df[, i]) == "factor") {
      if (length(levels(df[, i])) < 10)
        (next)() else stop("Error!! The number of categories should be less than 9")
    }
    
    ## make conresponding cutvec for validation_set: cut_vec_new
    cut_vec_new <- c(floor(min(df[, i])), cut_vec[[j]], ceiling(max(df[, i])))
    cut_vec_new_tmp <- signif(cut_vec_new, 3)
    cut_vec_new_tmp <- unique(cut_vec_new_tmp)  ###revised update##
    df[, i] <- cut(df[, i], breaks = cut_vec_new_tmp, right = F, include.lowest = T, dig.lab = 3)
    # xmin<-as.character(min(cut_vec_new_tmp)) xmax<-as.character(max(cut_vec_new_tmp))
    
    ## delete min and max for the Interval after discretion: validation_set
    df[, i]<-delete_min_max(df[, i])
    j <- j + 1
  }
  return(df)
}

delete_min_max<-function(vec){
  levels(vec)[1] <- gsub(".*,", "(,", levels(vec)[1])
  levels(vec)[length(levels(vec))] <- gsub(",.*", ",)", levels(vec)[length(levels(vec))])
  return(vec)
}

auto_test <- function(df, score_table) {
  for (i in 1:(length(names(df))-1)) {
    score_table_tmp <- score_table[grepl(names(df)[i], names(score_table))]
    df[, i] <- as.character(df[, i])
    for (j in 1:length(names(score_table_tmp))) {
      df[, i][df[, i] %in% gsub(names(df)[i], "", names(score_table_tmp)[j])] <- score_table_tmp[j]
    }
    
    df[, i] <- as.numeric(df[, i])
  }
  return(df)
}