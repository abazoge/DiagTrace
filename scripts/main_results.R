library(readxl)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
file_path  <- "../data/data.xlsx"
k_bonf     <- 5
output_dir <- "../tables"
dir.create(output_dir, showWarnings = FALSE)

model_pairs <- list(
  list(en = "o3_EN",         fr = "o3_FR",         name = "o3"),
  list(en = "DeepSeek_EN",   fr = "DeepSeek_FR",   name = "DeepSeek-R1"),
  list(en = "GPT_EN",        fr = "GPT_FR",        name = "GPT-4"),
  list(en = "Llama_EN",      fr = "Llama_FR",      name = "Llama-405B"),
  list(en = "Biomistral_EN", fr = "Biomistral_FR",  name = "BioMistral")
)

model_order <- c("o3", "DeepSeek-R1", "GPT-4", "Llama-405B", "BioMistral")

metrics <- list(
  list(col = "Note", max_score = 18, label = "Overall score (0-18)"),
  list(col = "D",    max_score = 3,  label = "Final diagnosis (0-3)"),
  list(col = "VI",   max_score = 5,  label = "Internal validity (0-5)"),
  list(col = "VE",   max_score = 3,  label = "External validity (0-3)"),
  list(col = "H",    max_score = 1,  label = "Differential diagnosis (0-1)"),
  list(col = "L",    max_score = 4,  label = "Logical structure (0-4)"),
  list(col = "E",    max_score = 2,  label = "Expression (0-2)")
)

metric_order <- sapply(metrics, function(m) m$label)

classification_sheets <- list(
  list(sheet = "CC_REF",  label = "Medical specialty",
       table_name = "Table 3", inferential = FALSE),
  list(sheet = "CC_RAIS", label = "Diagnostic reasoning type",
       table_name = "Table 4", inferential = TRUE),
  list(sheet = "CC_DIAG", label = "Diagnosis type",
       table_name = "Table 5", inferential = TRUE)
)

# ==============================================================================
# HELPERS
# ==============================================================================
get_sheet <- function(sheet_name, metric_col, path = file_path) {
  read_excel(path, sheet = sheet_name) %>%
    mutate(.row   = dplyr::row_number(),
           metric = suppressWarnings(as.numeric(.data[[metric_col]]))) %>%
    select(.row, ID, metric)
}

fmt_median_iqr <- function(x) {
  qs <- quantile(x, probs = c(0.25, 0.5, 0.75), na.rm = TRUE)
  sprintf("%.2f [%.2f;%.2f]", qs[["50%"]], qs[["25%"]], qs[["75%"]])
}

fmt_pct <- function(x, max_score) {
  sprintf("%.1f%%", 100 * mean(x == max_score, na.rm = TRUE))
}

fmt_p <- function(p) {
  if (is.na(p)) return("--")
  if (p < 0.001) return("<0.001")
  return(format.pval(p, digits = 3))
}

pair_by_row <- function(sheet_en, sheet_fr, metric_col, path = file_path) {
  df_en <- get_sheet(sheet_en, metric_col, path)
  df_fr <- get_sheet(sheet_fr, metric_col, path)
  paired <- inner_join(df_en, df_fr, by = ".row", suffix = c("_en", "_fr")) %>%
    filter(!is.na(metric_en), !is.na(metric_fr))
  paired$case_id <- paired$ID_en
  paired
}

run_lmm <- function(paired, k_bonf) {
  n_obs   <- nrow(paired)
  n_cases <- n_obs / 2

  df_long <- paired %>%
    mutate(rater_id = ifelse(row_number() <= n_cases, "rater1", "rater2")) %>%
    pivot_longer(cols = c(metric_en, metric_fr),
                 names_to  = "language",
                 values_to = "score") %>%
    mutate(language = factor(ifelse(language == "metric_en", "EN", "FR"),
                             levels = c("FR", "EN")),
           case_id  = factor(case_id),
           rater_id = factor(rater_id))

  fit <- lmer(score ~ language + (1 | case_id) + (1 | rater_id), data = df_long)
  s   <- summary(fit)
  lang_coef <- coef(s)["languageEN", ]

  estimate   <- lang_coef["Estimate"]
  p_twosided <- lang_coef["Pr(>|t|)"]
  p_onesided <- ifelse(estimate > 0, p_twosided / 2, 1 - p_twosided / 2)
  ci         <- confint(fit, parm = "languageEN", method = "Wald")


  # Residual normality check
  resids <- residuals(fit)
  sw <- shapiro.test(if (length(resids) > 5000) sample(resids, 5000) else resids)
  cat(sprintf("  Shapiro-Wilk: W = %.4f, p = %s\n",
              sw$statistic, format.pval(sw$p.value, digits = 3)))

  list(
    estimate = estimate,
    ci_str   = sprintf("[%.2f, %.2f]", ci[1], ci[2]),
    p_adj    = min(p_onesided * k_bonf, 1)
  )
}

# ==============================================================================
# TABLE 1: Overall score per model
# ==============================================================================
cat("Building Table 1...\n")

table1_rows <- list()
for (mp in model_pairs) {
  paired <- pair_by_row(mp$en, mp$fr, "Note")
  lmm <- tryCatch(run_lmm(paired, k_bonf),
                   error = function(e) list(estimate = NA, ci_str = "--", p_adj = NA))

  table1_rows[[length(table1_rows) + 1]] <- data.frame(
    Model           = mp$name,
    EN_median_IQR   = fmt_median_iqr(paired$metric_en),
    FR_median_IQR   = fmt_median_iqr(paired$metric_fr),
    Mean_diff       = ifelse(is.na(lmm$estimate), "--", sprintf("%.2f", lmm$estimate)),
    CI_95           = lmm$ci_str,
    Adj_P           = fmt_p(lmm$p_adj),
    stringsAsFactors = FALSE
  )
}

df_table1 <- do.call(rbind, table1_rows)
df_table1$Model <- factor(df_table1$Model, levels = model_order)
df_table1 <- df_table1 %>% arrange(Model)
names(df_table1) <- c("Model", "EN median [IQR]", "FR median [IQR]",
                       "Mean diff (LMM)", "95% CI", "Adj. P")

write.table(df_table1, file.path(output_dir, "table1.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("  Saved table1.tsv\n")

# ==============================================================================
# TABLE 2: Sub-scores grouped by criterion, then model
# ==============================================================================
cat("Building Table 2...\n")

table2_rows <- list()
for (m in metrics) {
  for (mp in model_pairs) {
    paired <- pair_by_row(mp$en, mp$fr, m$col)
    lmm <- tryCatch(run_lmm(paired, k_bonf),
                     error = function(e) list(estimate = NA, ci_str = "--", p_adj = NA))

    table2_rows[[length(table2_rows) + 1]] <- data.frame(
      Criterion      = m$label,
      Model          = mp$name,
      EN_median_IQR  = fmt_median_iqr(paired$metric_en),
      EN_pct_max     = fmt_pct(paired$metric_en, m$max_score),
      FR_median_IQR  = fmt_median_iqr(paired$metric_fr),
      FR_pct_max     = fmt_pct(paired$metric_fr, m$max_score),
      Mean_diff      = ifelse(is.na(lmm$estimate), "--", sprintf("%.2f", lmm$estimate)),
      CI_95          = lmm$ci_str,
      Adj_P          = fmt_p(lmm$p_adj),
      stringsAsFactors = FALSE
    )
  }
}

df_table2 <- do.call(rbind, table2_rows)
df_table2$Criterion <- factor(df_table2$Criterion, levels = metric_order)
df_table2$Model     <- factor(df_table2$Model, levels = model_order)
df_table2 <- df_table2 %>% arrange(Criterion, Model)
names(df_table2) <- c("Criterion (scale)", "Model", "EN median [IQR]", "% max EN",
                       "FR median [IQR]", "% max FR", "Mean diff", "95% CI", "Adj. P")

write.table(df_table2, file.path(output_dir, "table2.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("  Saved table2.tsv\n")

# ==============================================================================
# TABLES 3-5: By classification, grouped by category then model
# ==============================================================================
for (cls in classification_sheets) {
  cat(sprintf("Building %s...\n", cls$table_name))

  cc <- read_excel(file_path, sheet = cls$sheet) %>%
    select(ID, Classification)

  # Collect all data to determine category order (by total n descending)
  all_paired_list <- list()
  for (mp in model_pairs) {
    paired <- pair_by_row(mp$en, mp$fr, "Note")
    paired_cc <- paired %>%
      inner_join(cc, by = c("case_id" = "ID")) %>%
      filter(!is.na(Classification))
    paired_cc$model_name <- mp$name
    all_paired_list[[length(all_paired_list) + 1]] <- paired_cc
  }
  all_data <- do.call(rbind, all_paired_list)

  # Category order: by total n descending
  cat_order <- all_data %>%
    group_by(Classification) %>%
    summarise(total_n = n(), .groups = "drop") %>%
    arrange(desc(total_n)) %>%
    pull(Classification)

  cls_rows <- list()
  for (mp in model_pairs) {
    paired <- pair_by_row(mp$en, mp$fr, "Note")
    paired_cc <- paired %>%
      inner_join(cc, by = c("case_id" = "ID")) %>%
      filter(!is.na(Classification))

    categories <- paired_cc %>%
      group_by(Classification) %>%
      summarise(n = n(), .groups = "drop")

    for (i in seq_len(nrow(categories))) {
      cat_name <- categories$Classification[i]
      cat_n    <- categories$n[i]
      sub      <- paired_cc %>% filter(Classification == cat_name)

      row_data <- data.frame(
        Category       = cat_name,
        Model          = mp$name,
        n              = cat_n,
        EN_median_IQR  = fmt_median_iqr(sub$metric_en),
        EN_pct_max     = fmt_pct(sub$metric_en, 18),
        FR_median_IQR  = fmt_median_iqr(sub$metric_fr),
        FR_pct_max     = fmt_pct(sub$metric_fr, 18),
        stringsAsFactors = FALSE
      )

      if (cls$inferential) {
        lmm <- tryCatch(run_lmm(sub, k_bonf),
                         error = function(e) list(estimate = NA, ci_str = "--", p_adj = NA))
        row_data$Mean_diff <- ifelse(is.na(lmm$estimate), "--", sprintf("%.2f", lmm$estimate))
        row_data$CI_95     <- lmm$ci_str
        row_data$Adj_P     <- fmt_p(lmm$p_adj)
      }

      cls_rows[[length(cls_rows) + 1]] <- row_data
    }
  }

  df_cls <- do.call(rbind, cls_rows)

  # Reorder: category (by n desc) then model
  df_cls$Category <- factor(df_cls$Category, levels = cat_order)
  df_cls$Model    <- factor(df_cls$Model, levels = model_order)
  df_cls <- df_cls %>% arrange(Category, Model)

  if (cls$inferential) {
    names(df_cls) <- c(cls$label, "Model", "n", "EN median [IQR]", "% max EN",
                        "FR median [IQR]", "% max FR", "Mean diff", "95% CI", "Adj. P")
  } else {
    names(df_cls) <- c(cls$label, "Model", "n", "EN median [IQR]", "% max EN",
                        "FR median [IQR]", "% max FR")
  }

  # Determine filename from table name
  fname <- gsub(" ", "", tolower(cls$table_name))  # "table3", "table4", "table5"
  write.table(df_cls, file.path(output_dir, paste0(fname, ".tsv")),
              sep = "\t", row.names = FALSE, quote = FALSE)
  cat(sprintf("  Saved %s.tsv\n", fname))
}

# ==============================================================================
# PRINT ALL TABLES TO CONSOLE
# ==============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("TABLE 1 — Overall score (0-18)\n")
cat(strrep("=", 80), "\n")
write.table(df_table1, "", sep = "\t", row.names = FALSE, quote = FALSE)

cat("\n")
cat(strrep("=", 80), "\n")
cat("TABLE 2 — Sub-scores by criterion\n")
cat(strrep("=", 80), "\n")
write.table(df_table2, "", sep = "\t", row.names = FALSE, quote = FALSE)

# Print supplementary tables
for (cls in classification_sheets) {
  fname <- gsub(" ", "", tolower(cls$table_name))
  df_cls <- read.delim(file.path(output_dir, paste0(fname, ".tsv")),
                        sep = "\t", check.names = FALSE)
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat(sprintf("%s — %s\n", cls$table_name, cls$label))
  cat(strrep("=", 80), "\n")
  write.table(df_cls, "", sep = "\t", row.names = FALSE, quote = FALSE)
}

cat(sprintf("\nAll TSV files saved to: %s/\n", output_dir))