# API Reference

Complete API documentation for all chapkit modules, classes, and functions.

## Artifact Module

Hierarchical storage system for models, data, and experiment tracking.

### Models

::: chapkit.artifact.models

### Schemas

::: chapkit.artifact.schemas

### Repository

::: chapkit.artifact.repository

### Manager

::: chapkit.artifact.manager

### Router

::: chapkit.artifact.router

## Task Module

Reusable command templates for shell and Python task execution.

### Models

::: chapkit.task.models

### Schemas

::: chapkit.task.schemas

### Repository

::: chapkit.task.repository

### Manager

::: chapkit.task.manager

### Router

::: chapkit.task.router

### Registry

::: chapkit.task.registry

### Validation

::: chapkit.task.validation

## Config Module

Key-value configuration storage with Pydantic schema validation.

### Models

::: chapkit.config.models

### Schemas

::: chapkit.config.schemas

### Repository

::: chapkit.config.repository

### Manager

::: chapkit.config.manager

### Router

::: chapkit.config.router

## ML Module

Train/predict workflows with artifact-based model storage and timing metadata.

### Schemas

::: chapkit.ml.schemas

### Manager

::: chapkit.ml.manager

### Router

::: chapkit.ml.router

### Model Runners

Protocol and implementations for ML model training and prediction.

#### BaseModelRunner

::: chapkit.ml.runner.BaseModelRunner

#### FunctionalModelRunner

::: chapkit.ml.runner.FunctionalModelRunner

#### ShellModelRunner

::: chapkit.ml.runner.ShellModelRunner

## Scheduler

Chapkit job scheduler with artifact tracking for ML/task workflows.

::: chapkit.scheduler

## API Layer

FastAPI-specific components built on servicekit.

### Dependencies

FastAPI dependency injection functions.

::: chapkit.api.dependencies

### Alembic Helpers

Reusable migration helpers for chapkit database tables.

::: chapkit.alembic_helpers
