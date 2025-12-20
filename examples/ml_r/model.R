#!/usr/bin/env Rscript
#
# Mean Model for CHAP
#
# A simple example model that predicts future values based on historical means.
# Demonstrates integration with chapkit using chap.r.sdk.
#
# Usage (via chapkit):
#   Rscript model.R train --data data.csv --run-info run_info.yml
#   Rscript model.R predict --historic historic.csv --future future.csv --output predictions.csv --run-info run_info.yml
#   Rscript model.R info --format json
#

library(chap.r.sdk)
library(dplyr)
library(purrr)

# ============================================================================
# Model Configuration Schema
# ============================================================================
# This schema is exposed via the "info" command and used by chapkit
# for configuration validation and UI generation.

config_schema <- create_config_schema(
  title = "Mean Model Configuration",
  description = "Configuration for the simple mean prediction model",
  properties = list(
    smoothing = schema_number(
      description = "Smoothing factor for exponential smoothing (0 = no smoothing)",
      default = 0.0,
      minimum = 0.0,
      maximum = 1.0
    ),
    min_observations = schema_integer(
      description = "Minimum observations required per location",
      default = 1L,
      minimum = 1L
    )
  )
)

# ============================================================================
# Model Info (Service Metadata)
# ============================================================================
# Describes model requirements and capabilities for chapkit integration.

model_info <- list(
  period_type = "any",
  allows_additional_continuous_covariates = FALSE,
  required_covariates = character(0)
)

# ============================================================================
# Training Function
# ============================================================================
# Receives:
#   - training_data: tsibble with time series data
#   - model_configuration: list of config options (from config.yml)
#   - run_info: list with prediction_length, additional_covariates, etc.
#
# Returns:
#   - Model object (will be saved as RDS)

train_mean_model <- function(training_data, model_configuration = list(),
                              run_info = list()) {
  message("Training mean model...")

  # Extract config with defaults
  smoothing <- model_configuration$smoothing %||% 0.0
  min_obs <- model_configuration$min_observations %||% 1L

  message("  smoothing = ", smoothing)
  message("  min_observations = ", min_obs)

  # Calculate mean per location
  means <- training_data |>
    as_tibble() |>
    group_by(location) |>
    summarise(
      mean_cases = mean(disease_cases, na.rm = TRUE),
      n_obs = sum(!is.na(disease_cases)),
      .groups = "drop"
    ) |>
    filter(n_obs >= min_obs)

  message("  Computed means for ", nrow(means), " locations")

  # Return model object
  list(
    means = means,
    smoothing = smoothing,
    trained_at = Sys.time()
  )
}

# ============================================================================
# Prediction Function
# ============================================================================
# Receives:
#   - historic_data: tsibble with recent historical data
#   - future_data: tsibble with future time periods to predict
#   - saved_model: model object loaded from RDS
#   - model_configuration: list of config options
#   - run_info: list with prediction_length, additional_covariates, etc.
#
# Returns:
#   - tibble with 'samples' list-column containing prediction samples

predict_mean_model <- function(historic_data, future_data, saved_model,
                                model_configuration = list(), run_info = list()) {
  message("Generating predictions...")

  # Get prediction length from run_info
  pred_length <- run_info$prediction_length
  if (!is.null(pred_length) && !is.na(pred_length)) {
    message("  prediction_length = ", pred_length)
  }

  # Join means to future data
  predictions <- future_data |>
    as_tibble() |>
    left_join(saved_model$means, by = "location") |>
    mutate(
      # For locations without training data, use 0
      predicted_cases = coalesce(mean_cases, 0),
      # Create samples list-column (single sample for deterministic model)
      samples = map(predicted_cases, ~c(.x))
    ) |>
    select(-mean_cases, -n_obs, -predicted_cases)

  message("  Generated ", nrow(predictions), " predictions")

  predictions
}

# ============================================================================
# CLI Entry Point
# ============================================================================
# Uses chap.r.sdk's create_chapkit_cli() for chapkit-compatible interface.
# Supports subcommands: train, predict, info
#
# The info command with --format json outputs structured metadata for chapkit:
# {
#   "service_info": { "period_type": "any", ... },
#   "config_schema": { "$schema": "...", "properties": { ... } }
# }

if (!interactive()) {
  create_chapkit_cli(
    train_fn = train_mean_model,
    predict_fn = predict_mean_model,
    model_config_schema = config_schema,
    model_info = model_info
  )
}
