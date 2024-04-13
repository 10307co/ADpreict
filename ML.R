# 1.安装特征选择包Boruta
if(!requireNamespace('Boruta', quietly = TRUE)) {
  install.packages('Boruta')
}
library('Boruta')
library('caret') # 加载 caret 包用于数据预处理

# 2.理解数据
setwd("bingshengwang/desktop") # 设置工作路径，需要替换为您的实际路径
mydata <- read.csv("desktop/nacc_AD31.csv", sep = ",", header = TRUE) # 读取数据集

# 定义要进行分析的变量
continuous_vars <- c('EDUC', 'sdp', 'Age', 'Drugstaken', 'BMI')
categorical_vars <- c('SEX', 'Marital', 'Mom', 'Stenting', 'Hypertensive', 'Diabetes', 'DP', 'Neurosis', 'REM', 'Insomnia')

# 对连续变量进行数据预处理
preProcValues <- preProcess(mydata[, continuous_vars], method = c("center", "scale"))
mydata[, continuous_vars] <- predict(preProcValues, mydata[, continuous_vars])

# 分割数据为训练集和测试集
set.seed(42) # 设置随机数种子
trainIndex <- createDataPartition(mydata$AD, p = .7, list = FALSE, times = 1)
X_train <- mydata[trainIndex, ]
y_train <- mydata$AD[trainIndex]
X_test <- mydata[-trainIndex, ]
y_test <- mydata$AD[-trainIndex]

# 3.特征选择
set.seed(1) # 设置随机数种子
# 指定 Boruta 分析的变量
all_vars <- c(continuous_vars, categorical_vars)
formula_vars <- paste(all_vars, collapse = '+')
Boruta_formula <- as.formula(paste('AD ~', formula_vars))

# 应用 Boruta 特征选择
Boruta_result <- Boruta(Boruta_formula, data = X_train, doTrace = 0, pValue = 0.01, getImp = getImpRfZ,
                        mcAdj = TRUE, maxRuns = 100, holdHistory = TRUE)

# 绘制重要性历史图
plotImpHistory(Boruta_result, whichShadow = c(TRUE, TRUE, TRUE), ylab = "Z-Scores", ylim = c(0, 35), las = 2)
plot(Boruta_result, whichShadow = c(FALSE, FALSE, FALSE), xlab = "", ylab = "Importance: Z-Score", main = "Variable Importance", cex.axis = 0.6, ylim = c(0, 35), las = 2)
print(Boruta_result)

# 查看结构和最终决定
str(Boruta_result)
Boruta_result$finalDecision
Boruta_result[["finalDecision"]][["V1"]]
Boruta_result[["ImpHistory"]]

# 绘制重要性图
plot(Boruta_result, cex.axis = 0.65, las = 2, xlab = "", main = "Variable Importance")
plot(Boruta_result)

# 获取被Boruta确认的特征
confirmed_features <- getSelectedAttributes(Boruta_result, withTentative = FALSE)

# 使用caret包进行RFE特征选择
library(caret)

# 创建控制参数
ctrl <- rfeControl(functions = rfFuncs, method = "repeatedcv", number = 10, repeats = 3)

# 初始化随机森林模型
model <- randomForest(x = X_train[, confirmed_features], y = as.factor(y_train))

# 进行RFE
rfe_results <- rfe(x = X_train[, confirmed_features], y = as.factor(y_train), sizes = c(1:length(confirmed_features)),
                   rfeControl = ctrl, method = "rf")

# 打印出选择的最佳特征数量和特征名称
print(rfe_results)

# 查看RFE结果中选出的特征
optimal_features <- predictors(rfe_results)
print(optimal_features)


