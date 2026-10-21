# ============================================
# Step 14: draw the nomogram in R from the LR coefficients exported by Python
# The predictors are taken from the CSV written by step_13_lr_nomogram.py; nothing is hard-coded
# ============================================

rm(list = ls())

library(rms)

# 1. Read the coefficient file from Python (intercept, display names, ranges)
params <- read.csv("step_13_lr_nomogram_parameters.csv",
  stringsAsFactors = FALSE,
  check.names = FALSE,
  fileEncoding = "UTF-8-BOM"
)

# Separate the intercept row from the feature rows
intercept_row <- params[params$Feature == "Intercept", ]
feat_rows <- params[params$Feature != "Intercept", ]

coef_intercept <- intercept_row$Coefficient[1]

# Order the predictors by clinical domain:
# 1. demographics (Sex) -> 2. motor/medication (LEDD) -> 3. imaging (mean striatal SBR) ->
# 4. cognition (NP1COG, SVF) -> 5. other non-motor symptoms (SCOPA-AUT, RBDSQ score, UPSIT)
clinical_order <- c(
  "Sex (male=1; female=0)",
  "LEDD (mg/day)",
  "Mean striatal SBR (unitless)",
  "NP1COG",
  "SVF",
  "SCOPA-AUT",
  "RBDSQ total score",
  "UPSIT (percentile)"
)

# Reorder the feature rows
if (all(clinical_order %in% feat_rows$Label)) {
  feat_rows <- feat_rows[match(clinical_order, feat_rows$Label), ]
}

# Original names, display names, coefficients and ranges
raw_names <- feat_rows$Feature
disp_names <- feat_rows$Label
disp_names[raw_names == "upsit_pctl"] <- "UPSIT"
coefs <- feat_rows$Coefficient
mins <- feat_rows$Min_Value
maxs <- feat_rows$Max_Value

# 2. Read the original-scale training data
train_data <- read.csv("PPMI_4_train.csv", check.names = FALSE)

# Replace the original column names by display names
for (i in seq_along(raw_names)) {
  names(train_data)[names(train_data) == raw_names[i]] <- disp_names[i]
}

# Readable labels for binary sex
sex_label <- "Sex (male=1; female=0)"
if (sex_label %in% names(train_data)) {
  train_data[[sex_label]] <- factor(train_data[[sex_label]],
    levels = c(0, 1),
    labels = c("Female", "Male")
  )
}

# 3. Set the data distribution and force the axis ranges from the CSV
dd <- datadist(train_data)
for (i in seq_along(disp_names)) {
  if (!is.factor(train_data[[disp_names[i]]])) {
    dd$limits["Low:prediction", disp_names[i]] <- mins[i]
    dd$limits["High:prediction", disp_names[i]] <- maxs[i]
  }
}
options(datadist = "dd")

# 4. Build the model formula from the display names
formula_rhs <- paste(sprintf("`%s`", disp_names), collapse = " + ")
fit_formula <- as.formula(paste("Target ~", formula_rhs))

fit <- lrm(fit_formula, data = train_data, x = TRUE, y = TRUE)

# 5. Replace the R coefficients with those from Python (same order as the formula)
new_coef <- c(coef_intercept, coefs)
fit$coefficients[] <- unname(new_coef)

# 6. Check the theoretical range of predicted probabilities
#    highest risk: maximum for positive coefficients, minimum for negative ones
lp_max <- coef_intercept + sum(ifelse(coefs >= 0, coefs * maxs, coefs * mins))
lp_min <- coef_intercept + sum(ifelse(coefs >= 0, coefs * mins, coefs * maxs))
cat("Theoretical minimum risk =", plogis(lp_min), "\n")
cat("Theoretical maximum risk =", plogis(lp_max), "\n")

# 7. Readable clinical tick marks for each predictor (no spurious decimals or continuous ticks for discrete variables)
clean_ticks <- list(
  "SCOPA-AUT" = c(0, 10, 20, 30),
  "NP1COG" = c(0, 1, 2),
  "RBDSQ total score" = c(0, 3, 6, 9, 12),
  "SVF" = c(20, 35, 50, 65, 80),
  "Mean striatal SBR (unitless)" = c(0.2, 0.5, 0.8, 1.1, 1.4),
  "LEDD (mg/day)" = c(0, 500, 1000, 1500),
  "UPSIT" = c(10, 30, 50, 70, 90),
  "Age" = c(30, 40, 50, 60, 70, 80)
)

var_ticks <- list()
for (i in seq_along(disp_names)) {
  name_i <- disp_names[i]
  if (is.factor(train_data[[name_i]])) {
    next
  } else if (name_i %in% names(clean_ticks)) {
    candidate_ticks <- clean_ticks[[name_i]]
    candidate_ticks <- candidate_ticks[
      candidate_ticks >= mins[i] & candidate_ticks <= maxs[i]
    ]
    if (all(c(mins[i], maxs[i]) == round(c(mins[i], maxs[i])))) {
      endpoint_gap <- 0.15 * (maxs[i] - mins[i])
      candidate_ticks <- candidate_ticks[
        candidate_ticks > mins[i] + endpoint_gap &
          candidate_ticks < maxs[i] - endpoint_gap
      ]
      candidate_ticks <- c(mins[i], candidate_ticks, maxs[i])
    }
    var_ticks[[name_i]] <- sort(unique(candidate_ticks))
  } else if (maxs[i] - mins[i] <= 5 && all(train_data[[name_i]] == round(train_data[[name_i]]))) {
    var_ticks[[name_i]] <- seq(mins[i], maxs[i], by = 1)
  } else {
    candidate_ticks <- pretty(c(mins[i], maxs[i]), n = 5)
    endpoint_gap <- 0.08 * (maxs[i] - mins[i])
    var_ticks[[name_i]] <- sort(unique(c(
      mins[i],
      candidate_ticks[candidate_ticks > mins[i] + endpoint_gap &
        candidate_ticks < maxs[i] - endpoint_gap],
      maxs[i]
    )))
  }
}

prob_min <- plogis(lp_min)
prob_max <- plogis(lp_max)
candidate_prob_ticks <- c(
  0.02, 0.05, 0.1, 0.2, 0.4, 0.6,
  0.8, 0.9, 0.99
)
prob_ticks <- candidate_prob_ticks[
  candidate_prob_ticks > prob_min & candidate_prob_ticks < prob_max
]
prob_ticks <- prob_ticks[is.finite(qlogis(prob_ticks))]
if (length(prob_ticks) == 0) {
  stop("No finite probability ticks fall within the model prediction range")
}

nomo_args <- c(
  list(fit),
  var_ticks,
  list(
    lp.at     = c(lp_min, lp_max),
    fun       = plogis,
    fun.at    = prob_ticks,
    fun.lp.at = qlogis(prob_ticks),
    funlabel  = "Rapid-class probability",
    lp        = FALSE
  )
)

nomo <- do.call(nomogram, nomo_args)

# 8. Save the PDF
pdf_path <- "step_14_nomogram.pdf"

pdf(pdf_path, width = 12, height = 8)
par(lwd = 1.5)
plot(nomo,
  xfrac = 0.18, cex.var = 1.20, cex.axis = 1.00,
  lmgp = 0.35, label.every = 1, force.label = TRUE
)
dev.off()

# Crop the margins and overwrite step_14_nomogram.pdf (vector output)
if (Sys.which("pdfcrop") != "") {
  system2("pdfcrop", c(pdf_path, pdf_path))
}

cat("Nomogram saved (PDF).\n")
