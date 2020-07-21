library(data.table)
library(xgboost)
library(dplyr)
library(caTools)
library(DMwR)

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
# df_2015<-df %>% filter(substr(issue_d,5,8)>2015 & purpose %in% c('debt_consolidation','credit_card'))
df_1<-df %>% filter(substr(issue_d,5,8)==2015 & 
                        substr(last_pymnt_d,5,8)>2017 &
                        purpose %in% c('debt_consolidation','credit_card'))

table(df_1$loan_status)
df_1<-df_1[, -which(colMeans(is.na(df_1)) > 0.4)] # get rid of data that have 40% or above N/A 
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

var_y<-as.matrix(train_feature_full$y)
var_x<-subset(train_feature_full,select=-c(y))
dtrain1<-xgb.DMatrix(data.matrix(var_x),label=var_y)

param<-list(
  'objective' = "binary:logistic",
  'max_depth' = 5,
  'eta' = 0.1, 
  'eval_metric'='error',
  # 'nthread'=4,
  #gamma=2,
  'subsample'=0.8,
  # 'min_child_weight' = 5,
  'colsample_bytree'=0.8
)

set.seed(12)
bst1<-xgboost(params = param,data=dtrain1,nrounds = 300,verbose = 1,prediction=T)
names<-dimnames(var_x)[[2]]

importance1=xgb.importance(names,model=bst1)
importance1[,'cum_gain']<-cumsum(importance1[,'Gain'])

sel_var1<-importance1[importance1$cum_gain<=1,'Feature']

xtrain<-subset(train_feature_full[,c(sel_var1$Feature)])
xtest<-subset(test_feature_full[,c(sel_var1$Feature)])

train_full<-cbind(xtrain,y=train_feature_full$y)
test_full<-cbind(xtest,y=test_feature_full$y)

ytrain<-as.matrix(train_full$y)
ytest<-as.matrix(test_full$y)

#############build model######################
set.seed(123)
dtrain<-xgb.DMatrix(data.matrix(xtrain),label=ytrain)
param<-list(
  'objective' = "binary:logistic",
  'max_depth' = 3, 
  'eta' = 0.6, 
  'eval_metric'='error',
  'gamma'=3,
  'subsample'=0.7,
  'min_child_weight' =4,
  'colsample_bytree'=0.7
)

set.seed(12)
nround.cv=50
bst.cv<-xgb.cv(param=param,data=dtrain,nfold=10,nround=nround.cv,verbose=1,prediction=T)
min_error_idx <- which.min(bst.cv$evaluation_log$test_error_mean)
min_error_idx

bst<-xgboost(params=param,data=dtrain,nrounds = min_error_idx,verbose = 1,prediction=T)

##################results######################
xgb_train_pred<-predict(bst,data.matrix(xtrain))
KS(xgb_train_pred,ytrain)
AUC(xgb_train_pred,ytrain)

xgb_test_pred<-predict(bst,data.matrix(xtest))
KS(xgb_test_pred,ytest)
AUC(xgb_test_pred,ytest)

xgb_full_pred<-predict(bst,data.matrix(df_1[,unlist(sel_var1)]))

##################XGB confusion matrix#############
thres<-mean(train_feature_full$y)
train_pred_thres<-ifelse(xgb_pred_train>thres,1,0)
test_pred_thres<-ifelse(xgb_pred_test>thres,1,0)
confusionMatrix(factor(train_feature_full$y), factor(train_pred_thres))
confusionMatrix(factor(test_feature_full$y), factor(test_pred_thres))

#precision = Neg Pred Value 
#recall = specificity 
#F1-score = 2*(recall*precision)/(recall+precision) 

result <- confusionMatrix
precision <- result$byClass['Pos Pred Value']

##################gainstable######################
train_pred<-as.data.frame(cbind(train_full$y,xgb_train_pred))
names(train_pred)[1]<-'ytrain'
test_pred<-as.data.frame(cbind(test_full$y,xgb_test_pred))
names(test_pred)[1]<-'ytest'
train_10<-lift(train_pred$ytrain,train_pred$xgb_train_pred,groups=10)
test_10<-lift(test_pred$ytest,test_pred$xgb_test_pred,groups=10)

full<-lift(df_1$y,xgb_full_pred,groups=10)
names<-dimnames(xtrain)[[2]]
importance=xgb.importance(names,model=bst)
importance[,'cum_gain']=cumsum(importance[,'Gain'])
importance

df_2<-df %>% filter(substr(issue_d,5,8)==2015 & 
                      substr(last_pymnt_d,5,8)>2017 &
                      purpose %in% c('debt_consolidation','credit_card'))
bad2<-c("Charged Off", "Default")
df_2$loan_status<-ifelse(df_2$loan_status %in% bad2, 1,0)
df_2$pred<-xgb_full_pred
df_2[,'bin']<-ntile(desc(df_2[,'pred']),10)
df_3<-df_2[,c('loan_status','bin','out_prncp','pred')]
df_3$bad_os<-ifelse(df_3$loan_status==1,df_3$out_prncp,0)
df_3$good_os<-ifelse(df_3$loan_status==0,df_3$out_prncp,0)
df_3 %>% group_by(bin) %>% summarise(mean(pred))
df_out<-df_3 %>% group_by(bin) %>% summarise(loss = sum(bad_os), profit = sum(good_os)*0.12)

data.frame(df_out)
df_out2<-df_3 %>% group_by(bin) %>% summarise(max_pd=max(pred), min_pd=min(pred))
full_out<-cbind(full,df_out2)


# ############################oot#######################################
# df_oot<-df %>% filter(substr(issue_d,5,8)==2016 & 
#                         substr(last_pymnt_d,5,8)>2018 &
#                         purpose %in% c('debt_consolidation','credit_card'))
# 
# df_oot$loan_amnt<-log(df_oot$loan_amnt)
# df_oot$annual_inc<-log(df_oot$annual_inc)
# keepevent<-names(df_oot) %in% c('int_rate','term','dti','installment','annual_inc','loan_amnt'
#                               ,'sub_grade','grade','home_ownership','emp_length','percent_bc_gt_75'
#                               ,'verification_status','delinq_2yrs','loan_status')
# df_oot<-df_oot[keepevent]
# df_oot[is.na(df_oot)]<- -999
# df_oot<-df_oot%>% mutate_if(is.character,as.factor)
# bad<-c("Charged Off", "Default", "Late (31-120 days)")
# df_oot$loan_status<-ifelse(df_oot$loan_status %in% bad, 1,0)
# colnames(df_oot)[which(names(df_oot) == "loan_status")] <- "y"
# 
# xoot<-subset(df_oot[,c(sel_var1$Feature)])
# xgb_oot_pred<-predict(bst,data.matrix(xoot))
# KS(xgb_oot_pred,as.matrix(df_oot$y))
# AUC(xgb_oot_pred,as.matrix(df_oot$y))

#### TAYLOR ######
## Pie chart of the distribution of Loans across the different status:
sumAmnt(LC, loan_status_new) %>% merge(sumPerSatus(LC, loan_status_new)) %>%
  plot_ly(type = "pie", 
          labels = loan_status_new, 
          values = total_issued, 
          hole = 0.5,
          marker = list(colors = c("#f2f0f7","#dadaeb","#bcbddc","#9e9ac8","#807dba","#6a51a3","#4a1486"),
                        line = list(width = 1, color = "rgb(52, 110, 165)")),
          sort = F,
          direction = "counterclockwise",
          rotation = 90,
          textinfo = "label+percent",
          textfont = list(size = 14),
          text = paste("Default rates: ", charged),
          textposition = "outside") %>%
  layout(title = 'LOAN ISSUED GROUPED BY STATUS<br>(Hover for breakdown)',
         height = 500, width = 1400, autosize = T, 
         legend = list(font = list(size = 16), x = 1, y = 1, traceorder = "normal"))


# plot shows the distribution of fico scores of the borrowers against their LC's grade:
# although there is a trend, we can see on this plot that some borrowers with very high FICO scores
# got really poor LC Grade
library(extrafont)
qplot(grade, fico_range_high, data = df, geom = "boxplot", color = grade) +
  theme_economist() +
  scale_fill_economist() +
  xlab(toupper("LC Grades ranging from A to G")) +
  ylab("FICO SCORES") +
  ggtitle(toupper("Box plot of the FICO score distribution for each LC Grades"))

