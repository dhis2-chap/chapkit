# Chapkit Roadmap

This document outlines the development roadmap for Chapkit, tracking completed milestones and planned enhancements.

## Recently Completed (v0.2.0 - October 2024)

### Core Framework
- Migrated from PandasDataFrame to servicekit.data.DataFrame
- Improved test coverage from 90.92% to 93.43%
- Added comprehensive edge case testing for scheduler, task manager, and API dependencies
- Improved type checking configuration and pragma comments

### Examples & Testing
- Systematically tested all 12 examples (100% passing with `uv run python main.py`)
- Added Docker support for ml_pipeline and quickstart examples
- Fixed multiple worker issues with in-memory SQLite (set WORKERS=1 in all compose files)
- Gitignored example uv.lock files to reduce repository clutter
- Created comprehensive example testing report

### Documentation
- Updated CLAUDE.md with project guidelines
- Removed emojis from documentation for professional consistency
- Improved naming conventions (repository vs repo)

### Bug Fixes
- Fixed DataFrame import errors across codebase
- Fixed shell runner YAML config serialization
- Added missing continuous covariates to MLServiceInfo

## Near-term (v0.3.0)

### Testing & Quality
- [ ] Add CI/CD workflow to test all examples on each PR
- [ ] Create automated health check script for examples
- [ ] Improve coverage for remaining uncovered areas:
  - [ ] alembic_helpers.py (currently 0%)
  - [ ] artifact/manager.py edge cases
  - [ ] ml/runner.py additional scenarios
- [ ] Add integration tests for end-to-end ML workflows

### Documentation
- [ ] Add README.md to each example explaining what it demonstrates
- [ ] Create troubleshooting guide for common issues
- [ ] Document Docker deployment best practices
- [ ] Add API usage examples to documentation

### Infrastructure
- [ ] Pin servicekit to specific version once published to PyPI
- [ ] Consider production database options (file-based SQLite or PostgreSQL)
- [ ] Add database migration testing

## Medium-term (v0.4.0)

### Features
- [ ] Enhanced artifact operations:
  - [ ] Bulk artifact operations
  - [ ] Artifact search and filtering
  - [ ] Artifact metadata enrichment
- [ ] Config enhancements:
  - [ ] Config versioning
  - [ ] Config validation rules engine
  - [ ] Config diffing utilities
- [ ] ML workflow improvements:
  - [ ] Model comparison utilities
  - [ ] Experiment tracking dashboard
  - [ ] Model performance metrics storage

### Performance
- [ ] Optimize artifact tree building for large hierarchies
- [ ] Add caching layer for frequently accessed configs
- [ ] Profile and optimize database queries
- [ ] Add pagination for large result sets

### Testing
- [ ] Add performance benchmarks
- [ ] Create load testing scenarios
- [ ] Add property-based testing for critical paths

## Long-term (v0.5.0+)

### Advanced Features
- [ ] Multi-database support (PostgreSQL, MySQL)
- [ ] Distributed task execution
- [ ] Real-time ML inference endpoints
- [ ] Model registry with versioning
- [ ] A/B testing framework for models
- [ ] Data lineage tracking

### Extensibility
- [ ] Plugin system for custom ML runners
- [ ] Configurable storage backends for artifacts
- [ ] Custom validation hooks for configs
- [ ] Event system for workflow orchestration

### Developer Experience
- [ ] CLI tool for common operations
- [ ] Development templates/scaffolding
- [ ] Interactive debugging tools
- [ ] Performance profiling utilities

## Infrastructure & Operations

### Deployment
- [ ] Kubernetes deployment examples
- [ ] Docker Compose production setup
- [ ] Cloud deployment guides (AWS, GCP, Azure)
- [ ] Monitoring and observability setup

### Security
- [ ] Audit logging for all operations
- [ ] Secret management integration
- [ ] Role-based access control (RBAC)
- [ ] API rate limiting

### Reliability
- [ ] Backup and recovery procedures
- [ ] High availability configuration
- [ ] Disaster recovery planning
- [ ] Health check improvements

## Community & Ecosystem

- [ ] Contribution guidelines
- [ ] Code of conduct
- [ ] Public API stability guarantees
- [ ] Deprecation policy
- [ ] Release notes automation
- [ ] Integration examples with popular ML frameworks (PyTorch, TensorFlow, XGBoost)

## Technical Debt

- [ ] Refactor service_builder.py to reduce complexity
- [ ] Standardize error handling across modules
- [ ] Improve type hints coverage to 100%
- [ ] Consolidate duplicate test fixtures
- [ ] Remove deprecated code paths

## Notes

### Version Strategy
- **0.x.y**: API may change, focus on features and stability
- **1.0.0**: Stable API, production-ready, comprehensive testing
- **Post-1.0**: Semantic versioning with backward compatibility guarantees

### Prioritization Criteria
1. User-reported issues and bugs
2. Test coverage and quality improvements
3. Developer experience enhancements
4. Performance optimizations
5. New features based on user feedback

### How to Contribute

See completed items? Have suggestions? Open an issue or PR:
- Issues: https://github.com/dhis2-chap/chapkit/issues
- Pull Requests: https://github.com/dhis2-chap/chapkit/pulls

---

Last Updated: October 2024
