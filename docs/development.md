# PS57 Development Guidelines

## Git Workflow

Do not directly develop experimental features on `main`.

Use feature branches.

Examples:

- `feature/ai-model`
- `feature/sonar-processing`
- `feature/detection-intelligence`
- `feature/geospatial`
- `feature/frontend`
- `feature/backend`

## Commit Messages

Use clear commit messages.

Examples:

- `feat: add sonar preprocessing pipeline`
- `feat: add YOLO inference service`
- `feat: add detection confidence scoring`
- `fix: correct GPS coordinate conversion`
- `docs: update architecture`
- `test: add detection schema tests`

## Rules

1. Do not commit `venv/`.
2. Do not commit `node_modules/`.
3. Do not commit large datasets.
4. Do not commit trained model weights unless specifically required.
5. Test code before merging.
6. Keep modules independent.
7. Update documentation when architecture changes.
8. Do not change another member's module without discussing it.
9. Never commit passwords, API keys, or database credentials.
10. All production-facing outputs must follow the project's shared data schema.

## Pull Request Requirements

Before merging:

- Code runs locally.
- Tests pass.
- No secrets are included.
- Documentation is updated where necessary.
- API/data contracts remain compatible.