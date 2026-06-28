# Regression analysis: anxiety ~ alarms + NSI

# קריאת הנתונים
df <- read.csv("weekly_nsi_analysis.csv", check.names = FALSE)

# בדיקת שמות עמודות
print(names(df))

# העמודה הראשונה היא עמודת הזמן
time_col <- names(df)[1]

# המרה לתאריך אם אפשר
df[[time_col]] <- as.Date(df[[time_col]])

# לוודא שהמשתנים מספריים
df$alarms <- as.numeric(df$alarms)
df$NSI <- as.numeric(df$NSI)
df$anxiety <- as.numeric(df$anxiety)

# הסרת שורות חסרות
reg_df <- na.omit(df[, c(time_col, "alarms", "NSI", "anxiety")])

# פונקציית z-score
z_score <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}

# יצירת משתנים מתוקננים
reg_df$alarms_z <- z_score(reg_df$alarms)
reg_df$NSI_z <- z_score(reg_df$NSI)
reg_df$anxiety_z <- z_score(reg_df$anxiety)

# מודל רגרסיה לינארית מרובה
model <- lm(anxiety_z ~ alarms_z + NSI_z, data = reg_df)

# סיכום מלא של המודל
summary_model <- summary(model)
print(summary_model)

# טבלת מקדמים עבור נספח
coef_table <- as.data.frame(summary_model$coefficients)
coef_table$Variable <- rownames(coef_table)
rownames(coef_table) <- NULL

coef_table <- coef_table[, c("Variable", "Estimate", "Std. Error", "t value", "Pr(>|t|)")]

print(coef_table)

# רווחי סמך 95%
conf_intervals <- as.data.frame(confint(model))
conf_intervals$Variable <- rownames(conf_intervals)
rownames(conf_intervals) <- NULL

print(conf_intervals)

# מדדי התאמת מודל
model_fit <- data.frame(
  R_squared = summary_model$r.squared,
  Adjusted_R_squared = summary_model$adj.r.squared,
  F_statistic = summary_model$fstatistic[1],
  F_df1 = summary_model$fstatistic[2],
  F_df2 = summary_model$fstatistic[3],
  F_p_value = pf(
    summary_model$fstatistic[1],
    summary_model$fstatistic[2],
    summary_model$fstatistic[3],
    lower.tail = FALSE
  )
)

print(model_fit)

# קורלציה בין המשתנים המסבירים - לבדוק אם alarms ו-NSI קשורים מדי אחד לשני
predictor_correlation <- cor(reg_df$alarms_z, reg_df$NSI_z, use = "complete.obs")
print(paste("Correlation between alarms and NSI:", round(predictor_correlation, 3)))

# אבחון שאריות בסיסי
reg_df$fitted <- fitted(model)
reg_df$residuals <- residuals(model)

# גרף שאריות מול ערכים חזויים (נשמר כקובץ)
png("regression_residuals_vs_fitted.png", width = 900, height = 600)
plot(
  reg_df$fitted,
  reg_df$residuals,
  xlab = "Fitted values",
  ylab = "Residuals",
  main = "Residuals vs Fitted Values"
)
abline(h = 0, col = "red", lty = 2)
dev.off()

# QQ plot לשאריות (נשמר כקובץ)
png("regression_qq_plot.png", width = 900, height = 600)
qqnorm(reg_df$residuals)
qqline(reg_df$residuals, col = "red")
dev.off()

# שמירת טבלאות לנספח
write.csv(coef_table, "regression_coefficients_appendix.csv", row.names = FALSE)
write.csv(conf_intervals, "regression_confidence_intervals_appendix.csv", row.names = FALSE)
write.csv(model_fit, "regression_model_fit_appendix.csv", row.names = FALSE)

cat("Saved appendix files:\n")
cat("regression_coefficients_appendix.csv\n")
cat("regression_confidence_intervals_appendix.csv\n")
cat("regression_model_fit_appendix.csv\n")