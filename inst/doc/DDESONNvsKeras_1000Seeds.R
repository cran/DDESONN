## ----setup, message=FALSE, warning=FALSE--------------------------------------
suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(knitr)
})

if (!requireNamespace("DDESONN", quietly = TRUE)) {
  message("DDESONN not installed in this build session; skipping evaluation.")
  knitr::opts_chunk$set(eval = FALSE)
}

## ----helpers, message=FALSE, warning=FALSE------------------------------------
.render_tbl <- function(x, title = NULL, digits = 4) {
  if (requireNamespace("DDESONN", quietly = TRUE) &&
      exists("ddesonn_viewTables", envir = asNamespace("DDESONN"), inherits = FALSE)) {
    get("ddesonn_viewTables", envir = asNamespace("DDESONN"))(x, title = title)
  } else {
    if (!is.null(title)) cat("\n\n###", title, "\n\n")
    knitr::kable(x, digits = digits, format = "html")
  }
}

## ----ddesonn-summary, message=FALSE, warning=FALSE, results='asis'------------
heart_failure_root <- system.file("extdata", "heart_failure_runs", package = "DDESONN")

if (!nzchar(heart_failure_root)) {
  # Fallback when building from source before installation
  heart_failure_root <- file.path("..", "inst", "extdata", "heart_failure_runs")
}

stopifnot(dir.exists(heart_failure_root))

train_run1_path <- file.path(
  heart_failure_root, "run1",
  "SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025.rds"
)
test_run1_path <- file.path(
  heart_failure_root, "run1",
  "SingleRun_Test_Metrics_500_seeds_20251025.rds"
)
train_run2_path <- file.path(
  heart_failure_root, "run2",
  "SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026.rds"
)
test_run2_path <- file.path(
  heart_failure_root, "run2",
  "SingleRun_Test_Metrics_500_seeds_20251026.rds"
)

stopifnot(
  file.exists(train_run1_path),
  file.exists(test_run1_path),
  file.exists(train_run2_path),
  file.exists(test_run2_path)
)

train_run1 <- readRDS(train_run1_path)
test_run1  <- readRDS(test_run1_path)
train_run2 <- readRDS(train_run2_path)
test_run2  <- readRDS(test_run2_path)

train_all <- dplyr::bind_rows(train_run1, train_run2)
test_all  <- dplyr::bind_rows(test_run1, test_run2)

train_seed <- train_all %>%
  group_by(seed) %>%
  slice_max(order_by = best_val_acc, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed,
    train_acc = best_train_acc,
    val_acc   = best_val_acc
  )

test_seed <- test_all %>%
  group_by(seed) %>%
  slice_max(order_by = accuracy, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed,
    test_acc = accuracy
  )

merged <- inner_join(train_seed, test_seed, by = "seed") %>%
  arrange(seed)

summarize_column <- function(x) {
  pct <- function(p) stats::quantile(x, probs = p, names = FALSE, type = 7)
  data.frame(
    count = length(x),
    mean  = mean(x),
    std   = sd(x),
    min   = min(x),
    `25%` = pct(0.25),
    `50%` = pct(0.50),
    `75%` = pct(0.75),
    max   = max(x),
    check.names = FALSE
  )
}

summary_train <- summarize_column(merged$train_acc)
summary_val   <- summarize_column(merged$val_acc)
summary_test  <- summarize_column(merged$test_acc)

summary_all <- data.frame(
  stat = c("count","mean","std","min","25%","50%","75%","max"),
  train_acc = unlist(summary_train[1, ]),
  val_acc   = unlist(summary_val[1, ]),
  test_acc  = unlist(summary_test[1, ]),
  check.names = FALSE
)

round4 <- function(x) if (is.numeric(x)) round(x, 4) else x
pretty_summary <- as.data.frame(lapply(summary_all, round4))

.render_tbl(
  pretty_summary,
  title = "DDESONN — 1000-seed summary (train/val/test)"
)

## ----keras-summary, message=FALSE, warning=FALSE, results='asis'--------------
if (!requireNamespace("readxl", quietly = TRUE)) {
  message("Skipping keras-summary chunk: 'readxl' not installed.")
} else {
  keras_path <- system.file(
    "scripts", "vsKeras", "1000SEEDSRESULTSvsKeras", "1000seedsKeras.xlsx",
    package = "DDESONN"
  )

  if (nzchar(keras_path) && file.exists(keras_path)) {
    keras_stats <- readxl::read_excel(keras_path, sheet = 2)
    .render_tbl(
      keras_stats,
      title = "Keras — 1000-seed summary (Sheet 2)"
    )
  } else {
    cat("Keras Excel not found in installed package.\n")
  }
}

## ----benchmark-comparison, message=FALSE, warning=FALSE, results='asis'-------
benchmark_results <- data.frame(
  Metric = c(
    "Mean Test Accuracy",
    "Standard Deviation",
    "Minimum Test Accuracy",
    "Maximum Test Accuracy"
  ),
  DDESONN = c("≈ 99.92%", "≈ 0.0013", "≈ 99.20%", "100%"),
  Keras   = c("≈ 99.69%", "≈ 0.0036", "≈ 97.82%", "100%"),
  check.names = FALSE
)

.render_tbl(
  benchmark_results,
  title = "Benchmark results across 1000 seeds"
)

