library(readxl)
library(dplyr)
library(tidyr)

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

# ==============================================================================
# AGGREGATE: mean of 2 raters per case
# ==============================================================================
aggregate_raters <- function(paired) {
  n_cases <- nrow(paired) / 2
  paired %>%
    mutate(case_idx = ifelse(row_number() <= n_cases,
                             row_number(),
                             row_number() - n_cases)) %>%
    group_by(case_idx) %>%
    summarise(metric_en = mean(metric_en, na.rm = TRUE),
              metric_fr = mean(metric_fr, na.rm = TRUE),
              .groups = "drop")
}

# ==============================================================================
# SENSITIVITY ANALYSIS: Overall score (0-18)
# ==============================================================================
cat("Building sensitivity analysis table (overall score)...\n")

rows_overall <- list()
for (mp in model_pairs) {
  paired <- pair_by_row(mp$en, mp$fr, "Note")
  paired_agg <- aggregate_raters(paired)

  x <- paired_agg$metric_en
  y <- paired_agg$metric_fr

  t_res <- wilcox.test(x, y, paired = TRUE, alternative = "greater",
                       conf.int = TRUE, exact = FALSE)

  # Effect size: rank-biserial correlation
  diffs <- x - y
  n_pos <- sum(diffs > 0, na.rm = TRUE)
  n_neg <- sum(diffs < 0, na.rm = TRUE)
  n_tie <- sum(diffs == 0, na.rm = TRUE)
  r_rb  <- ifelse((n_pos + n_neg) > 0, (n_pos - n_neg) / (n_pos + n_neg), 0)

  rows_overall[[length(rows_overall) + 1]] <- data.frame(
    Model           = mp$name,
    n               = nrow(paired_agg),
    EN_median_IQR   = fmt_median_iqr(x),
    FR_median_IQR   = fmt_median_iqr(y),
    V               = formatC(t_res$statistic, format = "d"),
    HL_estimate     = sprintf("%.2f", t_res$estimate),
    CI_95           = sprintf("[%.2f, Inf]", t_res$conf.int[1]),
    r_rb            = sprintf("%.2f", r_rb),
    EN_gt_FR        = n_pos,
    FR_gt_EN        = n_neg,
    Ties            = n_tie,
    P_value         = fmt_p(t_res$p.value),
    Adj_P           = fmt_p(min(t_res$p.value * k_bonf, 1)),
    stringsAsFactors = FALSE
  )
  cat(sprintf("  %s: done\n", mp$name))
}

df_overall <- do.call(rbind, rows_overall)
df_overall$Model <- factor(df_overall$Model, levels = model_order)
df_overall <- df_overall %>% arrange(Model)
names(df_overall) <- c("Model", "n", "EN median [IQR]", "FR median [IQR]",
                        "V", "Hodges-Lehmann", "95% CI", "Rank-biserial r",
                        "EN > FR", "FR > EN", "Ties", "P", "Adj. P")

write.table(df_overall, file.path(output_dir, "sensitivity_overall.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("  Saved sensitivity_overall.tsv\n")

# ==============================================================================
# SENSITIVITY ANALYSIS: Per sub-score
# ==============================================================================
cat("Building sensitivity analysis table (sub-scores)...\n")

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

rows_sub <- list()
for (m in metrics) {
  for (mp in model_pairs) {
    paired <- pair_by_row(mp$en, mp$fr, m$col)
    paired_agg <- aggregate_raters(paired)

    x <- paired_agg$metric_en
    y <- paired_agg$metric_fr

    t_res <- tryCatch({
      wilcox.test(x, y, paired = TRUE, alternative = "greater",
                  conf.int = TRUE, exact = FALSE)
    }, error = function(e) NULL)

    if (is.null(t_res)) {
      rows_sub[[length(rows_sub) + 1]] <- data.frame(
        Criterion       = m$label,
        Model           = mp$name,
        n               = nrow(paired_agg),
        EN_median_IQR   = fmt_median_iqr(x),
        FR_median_IQR   = fmt_median_iqr(y),
        V               = "--",
        HL_estimate     = "--",
        P_value         = "--",
        Adj_P           = "--",
        stringsAsFactors = FALSE
      )
    } else {
      rows_sub[[length(rows_sub) + 1]] <- data.frame(
        Criterion       = m$label,
        Model           = mp$name,
        n               = nrow(paired_agg),
        EN_median_IQR   = fmt_median_iqr(x),
        FR_median_IQR   = fmt_median_iqr(y),
        V               = formatC(t_res$statistic, format = "d"),
        HL_estimate     = sprintf("%.2f", t_res$estimate),
        P_value         = fmt_p(t_res$p.value),
        Adj_P           = fmt_p(min(t_res$p.value * k_bonf, 1)),
        stringsAsFactors = FALSE
      )
    }
  }
}

df_sub <- do.call(rbind, rows_sub)
df_sub$Criterion <- factor(df_sub$Criterion, levels = metric_order)
df_sub$Model     <- factor(df_sub$Model, levels = model_order)
df_sub <- df_sub %>% arrange(Criterion, Model)
names(df_sub) <- c("Criterion (scale)", "Model", "n", "EN median [IQR]",
                    "FR median [IQR]", "V", "Hodges-Lehmann", "P", "Adj. P")

write.table(df_sub, file.path(output_dir, "sensitivity_subscores.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("  Saved sensitivity_subscores.tsv\n")

# ==============================================================================
# PRINT TO CONSOLE
# ==============================================================================
cat("\n")
cat(strrep("=", 80), "\n")
cat("SENSITIVITY ANALYSIS — Overall score (aggregated Wilcoxon, n=180)\n")
cat(strrep("=", 80), "\n")
write.table(df_overall, "", sep = "\t", row.names = FALSE, quote = FALSE)

cat("\n")
cat(strrep("=", 80), "\n")
cat("SENSITIVITY ANALYSIS — Sub-scores (aggregated Wilcoxon, n=180)\n")
cat(strrep("=", 80), "\n")
write.table(df_sub, "", sep = "\t", row.names = FALSE, quote = FALSE)

cat(sprintf("\nTSV files saved to: %s/\n", output_dir))