library(data.table)
library(dplyr)
library(caTools)
library(DMwR)
library(glmnet)

AUC<-function(pred,depvar){
  require(ROCR)
  p<-prediction(as.numeric(pred),depvar)
  auc<-performance(p,'auc')
  auc<-unlist(slot(auc,'y.values'))
  return(auc)
}

KS<-function(pred,depvar){
  require(ROCR)
  p<-prediction(as.numeric(pred),depvar)
  perf<-performance(p,'tpr','fpr')
  ks<-max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
  return(ks)
}

lift <- function(depvar,predcol,groups){
  if(is.factor(depvar)){depvar<-as.integer(as.character(depvar))}
  if(is.factor(predcol)){predcol<-as.integer(as.character(predcol))}
  helper=data.frame(cbind(depvar,predcol))
  helper[,'bucket']=ntile(desc(helper[,'predcol']),groups)
  gaintable=helper %>% group_by(bucket) %>% 
    summarise_at(vars(depvar),funs(total=n(),
                                   totalresp=sum(.,na.rm=TRUE))) %>% 
    mutate(HitRate=totalresp/total*100,
           Cumresp=cumsum(totalresp),
           Gain=Cumresp/sum(totalresp)*100,
           Cumlift=Gain/(bucket*(100/groups)))
  return(gaintable)
}

df<-fread("/loan.csv",data.table = F)
table(substr(df$issue_d,5,8) )
table(substr(df$last_pymnt_d,5,8) )
chk<-df[which(df$loan_status %in% bad),'out_prncp']

# df_2015<-df %>% filter(substr(issue_d,5,8)>2015 & purpose %in% c('debt_consolidation','credit_card'))
df_1<-df %>% filter(substr(issue_d,5,8)==2015 & 
                      substr(last_pymnt_d,5,8)>2017 &
                      purpose %in% c('debt_consolidation','credit_card'))

table(df_1$loan_status)
df_1<-df_1[, -which(colMeans(is.na(df_1)) > 0.4)]
df_1$loan_amnt<-log(df_1$loan_amnt)
df_1$annual_inc<-log(df_1$annual_inc)
keepevent<-names(df_1) %in% c('int_rate','term','dti','installment','annual_inc','loan_amnt'
                              ,'sub_grade','grade','home_ownership','emp_length'
                              ,'verification_status','loan_status')

df_1<-df_1[keepevent]

names(which(colSums(is.na(df_1))>0))
names(which(colSums(df_1=='')>0))

df_1[is.na(df_1)]<- -999

df_1<-df_1%>% mutate_if(is.character,as.factor)

bad<-c("Charged Off", "Default", "Late (31-120 days)")
# bad<-c("Charged Off", "Default")
df_1$loan_status<-ifelse(df_1$loan_status %in% bad, 1,0)

colnames(df_1)[which(names(df_1) == "loan_status")] <- "y"
table(df_1$y)/nrow(df_1) 


set.seed(12)
split=sample.split(df_1$y,SplitRatio = 0.7)
train_feature_full<-subset(df_1,split==TRUE)
test_feature_full<-subset(df_1,split==FALSE)

x_train<-model.matrix(y~.,train_feature_full)[,-1]
y_train<-train_feature_full$y
x_test<-model.matrix(y~.,test_feature_full)[,-1]
y_test<-test_feature_full$y



# #logit
f<-formula(paste0('y','~',paste0(dimnames(train_feature_full)[[2]][-11],collapse="+")))
glm.fit = glm(f,family = "binomial",data=train_feature_full)
glm_train_pred=predict(glm.fit,type="response")
glm_test_pred=predict(glm.fit,test_feature_full,type="response")
summary(glm.fit)

#ridge
set.seed(12)
grid=10^seq(10,-2,length=100)
ridge.mod=glmnet(x_train,y_train,alpha=0,lambda=grid)
cv.out1=cv.glmnet(x_train,y_train,alpha=0,type.measure="mse")
# plot(cv.out1)
ridge_pred_train=predict(ridge.mod,s=cv.out1$lambda.min,newx=x_train,type = "response")
ridge_pred_test=predict(ridge.mod,s=cv.out1$lambda.min,newx=x_test,type = "response")
out_r=glmnet(x_train,y_train,alpha=0)
predict(out_r,type="coefficients",s=cv.out1$lambda.min)

#lasso
lasso.mod=glmnet(x_train,y_train,alpha=1,lambda=grid)
# plot(lasso.mod)
cv.out2=cv.glmnet(x_train,y_train,alpha=1,type.measure="mse")
# plot(cv.out2)
lasso_pred_train=predict(lasso.mod,s=cv.out2$lambda.min,newx=x_train,type = "response")
lasso_pred_test=predict(lasso.mod,s=cv.out2$lambda.min,newx=x_test,type = "response")

##################results######################
#logit regression
AUC(glm_train_pred,train_feature_full$y)
AUC(glm_test_pred,test_feature_full$y)
#ridge
AUC(ridge_pred_train,y_train)
AUC(ridge_pred_test,y_test)
#lasso
AUC(lasso_pred_train,y_train)
AUC(lasso_pred_test,y_test)
##################ridge features
out=glmnet(x_train,y_train,alpha=0)
predict(out,type="coefficients",s=cv.out1$lambda.min)
##################lasso features
out=glmnet(x_train,y_train,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=cv.out2$lambda.min)
lasso.coef
lasso.coef[lasso.coef!=0]

## 
#logit 
AUC(glm_train_pred,train_feature_full$y) - AUC(glm_test_pred,test_feature_full$y)
#ridge 
AUC(ridge_pred_train,y_train) - AUC(ridge_pred_test,y_test)
#lasso
AUC(lasso_pred_train,y_train) - AUC(lasso_pred_test,y_test)


#######confusion matrix###########

#logit
thres<-mean(train_feature_full$y)
train_pred_thres<-ifelse(glm_train_pred>thres,1,0)
test_pred_thres<-ifelse(glm_test_pred>thres,1,0)
confusionMatrix(factor(train_feature_full$y), factor(train_pred_thres))
confusionMatrix(factor(test_feature_full$y), factor(test_pred_thres))

#ridge 
thres<-mean(train_feature_full$y)
train_pred_thres<-ifelse(ridge_pred_train>thres,1,0)
test_pred_thres<-ifelse(ridge_pred_test>thres,1,0)
confusionMatrix(factor(train_feature_full$y), factor(train_pred_thres))
confusionMatrix(factor(test_feature_full$y), factor(test_pred_thres))

#lasso 
thres<-mean(train_feature_full$y)
train_pred_thres<-ifelse(lasso_pred_train>thres,1,0)
test_pred_thres<-ifelse(lasso_pred_test>thres,1,0)
confusionMatrix(factor(train_feature_full$y), factor(train_pred_thres))
confusionMatrix(factor(test_feature_full$y), factor(test_pred_thres))


###### ROC curve
roc_rose<-NULL

library(pROC)
roc_rose <- plot(roc(test_feature_full$y, glm_test_pred), print.auc = TRUE, col = "blue")

roc_rose <- plot(roc(y_test, ridge_pred_test), print.auc = TRUE, 
                 col = "green", print.auc.y = .4, add = TRUE)

roc_rose <- plot(roc(y_test, lasso_pred_test), print.auc = TRUE, 
                 col = "red", print.auc.y = .3, add = TRUE)

roc_rose <- plot(roc(test_full$y, xgb_test_pred), print.auc = TRUE, 
                 col = "black", print.auc.y = .2, add = TRUE)

legend("bottomright", legend=c("logit", "ridge", "lasso","XGB"),
       col=c("blue","green","red","black"), lwd=1)


### ROC plot for each model 
library(pROC)
#logit 
plot(roc(test_feature_full$y, glm_test_pred, direction="<"), col="red", lwd=3, print.auc = TRUE, main="Logistic Regression")

#ridge 
plot(roc(y_test, ridge_pred_test, direction="<"), col="red", lwd=3, print.auc = TRUE, main="Ridge Regression")

#lasso
plot(roc(y_test, lasso_pred_test, direction="<"), col="red", lwd=3, print.auc = TRUE, main="Lasso Regression")

#XGB # need a windows computer to plot ottherwise the AUC is low 
plot(roc(test_full$y, xgb_test_pred, direction="<"), col="red", lwd=3, print.auc = TRUE, main="XGBoost")

##### importance 
#logit
train_pred<-as.data.frame(cbind(train_feature_full$y,glm_train_pred))
names(train_pred)[1]<-'ytrain'
test_pred<-as.data.frame(cbind(test_feature_full$y,glm_test_pred))
names(test_pred)[1]<-'ytest'
train_10<-lift(train_pred$ytrain,train_pred$glm_train_pred,groups=10)
test_10<-lift(test_pred$ytest,test_pred$glm_test_pred,groups=10)

names<-dimnames(xtrain)[[2]]
importance=xgb.importance(names,model=bst)
importance[,'cum_gain']=cumsum(importance[,'Gain'])
importance












