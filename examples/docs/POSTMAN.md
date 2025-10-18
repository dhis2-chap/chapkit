# Postman Collections for Chapkit Examples

This directory contains Postman Collection v2.1 JSON files that you can import directly into Postman to test Chapkit services.

## Available Collections

### Authentication Collections

#### auth_basic.postman_collection.json
**API Key Authentication - Complete Workflow**
- Environment variable auth (CHAPKIT_API_KEYS)
- Multiple API keys for rotation
- Authentication failures (missing/invalid keys)
- Full CRUD workflow with auth
- RFC 9457 error responses

### ML Collections

#### ml_basic.postman_collection.json
**Disease Prediction - Functional Runner**
- Simple functional programming style
- 2 features: rainfall, mean_temperature
- LinearRegression model
- Status: Yellow (ready for testing)

#### ml_class.postman_collection.json
**Weather Prediction - Class-Based Runner**
- Object-oriented programming style
- 3 features: rainfall, mean_temperature, humidity
- LinearRegression with StandardScaler preprocessing
- Lifecycle hooks (on_init, on_cleanup)
- Status: Green (production-ready)

#### ml_shell.postman_collection.json
**Shell-Based - Language-Agnostic**
- External script execution
- 2 features: rainfall, mean_temperature
- Language-agnostic (Python, R, Julia, Docker)
- File-based data interchange
- Status: Orange (needs evaluation)

## How to Import

### Option 1: Import via Postman UI

1. Open Postman
2. Click **Import** button (top-left)
3. Select **Upload Files** tab
4. Choose one or more `.postman_collection.json` files
5. Click **Import**

### Option 2: Import via URL (if hosted)

1. Open Postman
2. Click **Import** → **Link**
3. Paste the raw GitHub URL to the JSON file
4. Click **Continue** → **Import**

### Option 3: Drag and Drop

1. Open Postman
2. Drag the `.postman_collection.json` file into the Postman window
3. Collection will be imported automatically

## Collection Structure

Each collection is organized into folders:

```
Collection
├── 1. Service Health & Info
│   ├── Check Service Health
│   ├── View Service Info
│   └── Get Config Schema
├── 2. Configuration Management
│   ├── Create Configuration
│   ├── List All Configs
│   └── Get Config by ID
├── 3. Model Training
│   ├── Train Model
│   ├── Poll Training Job
│   └── Get Trained Model Artifact
├── 4. Predictions
│   ├── Make Predictions
│   ├── Poll Prediction Job
│   └── Get Predictions
├── 5. Job Management
│   └── List/Filter/Cancel Jobs
└── 6. Artifact Management
    └── List/Delete Artifacts
```

## Collection Variables

Each collection includes pre-configured variables:

| Variable | Default Value | Description |
|----------|--------------|-------------|
| `baseUrl` | `http://127.0.0.1:8000` | API base URL |
| `config_id` | (auto-set) | Configuration ID |
| `train_job_id` | (auto-set) | Training job ID |
| `model_artifact_id` | (auto-set) | Model artifact ID |
| `predict_job_id` | (auto-set) | Prediction job ID |
| `prediction_artifact_id` | (auto-set) | Prediction artifact ID |

**Auto-set variables:** Test scripts automatically capture IDs from responses and store them in collection variables for subsequent requests.

## Usage Workflow

### Authentication Workflow (auth_basic)

1. **Start the service:**
   ```bash
   export CHAPKIT_API_KEYS="sk_dev_abc123,sk_dev_xyz789"
   fastapi dev examples/auth_basic.py
   ```

2. **Check service health (no auth):**
   - Run: `1. Service Health & Info` → `Check Service Health`
   - Verify: Status 200, `"status": "healthy"`

3. **Test authentication failures:**
   - Run: `2. Authentication Failures` → `Access Without Auth`
   - Verify: Status 401, missing auth header error
   - Run: `2. Authentication Failures` → `Access With Invalid Key`
   - Verify: Status 401, invalid key error

4. **Create configuration (with auth):**
   - Run: `3. Configuration Management` → `Create Configuration`
   - Auto-captured: `config_id` variable
   - Uses: `X-API-Key: {{api_key}}` header

5. **Perform CRUD operations:**
   - Run other requests in `3. Configuration Management`
   - All automatically use stored `config_id`

6. **Test key rotation:**
   - Run: `4. Key Rotation` → `Access With First Key`
   - Run: `4. Key Rotation` → `Access With Second Key`
   - Verify: Both keys work (zero-downtime rotation)

### ML Workflow (All Collections)

1. **Start the service:**
   ```bash
   fastapi dev examples/ml_basic.py
   # or ml_class.py, ml_shell.py
   ```

2. **Check service health:**
   - Run: `1. Service Health & Info` → `Check Service Health`
   - Verify: Status 200, `"status": "healthy"`

3. **Create configuration:**
   - Run: `2. Configuration Management` → `Create Configuration`
   - Auto-captured: `config_id` variable

4. **Train model:**
   - Run: `3. Model Training` → `Train Model`
   - Auto-captured: `train_job_id`, `model_artifact_id`

5. **Poll training job:**
   - Run: `3. Model Training` → `Poll Training Job`
   - Repeat until: `"status": "completed"`

6. **Make predictions:**
   - Run: `4. Predictions` → `Make Predictions`
   - Auto-captured: `predict_job_id`, `prediction_artifact_id`

7. **Poll prediction job:**
   - Run: `4. Predictions` → `Poll Prediction Job`
   - Repeat until: `"status": "completed"`

8. **Get predictions:**
   - Run: `4. Predictions` → `Get Predictions`
   - View results in `sample_0` column

### Running the Complete Workflow

Use Postman's **Collection Runner** to execute all requests in sequence:

1. Right-click collection → **Run collection**
2. Select requests to run (or run all)
3. Set iterations: 1
4. Set delay: 2000ms (for job polling)
5. Click **Run**

**Note:** Collection Runner doesn't automatically wait for async jobs to complete. For complete automation, use the manual workflow above with polling.

## Example Requests

### ml_basic Collection

**Train with 8 samples:**
```json
POST /api/v1/ml/$train
{
  "config_id": "{{config_id}}",
  "data": {
    "columns": ["rainfall", "mean_temperature", "disease_cases"],
    "data": [
      [100.5, 25.3, 12],
      [85.2, 27.1, 8],
      [120.8, 24.5, 15],
      [95.3, 26.2, 10],
      [110.0, 26.0, 13],
      [75.5, 28.0, 6],
      [130.2, 23.8, 18],
      [88.9, 27.5, 9]
    ]
  }
}
```

**Predict with 4 samples:**
```json
POST /api/v1/ml/$predict
{
  "model_artifact_id": "{{model_artifact_id}}",
  "future": {
    "columns": ["rainfall", "mean_temperature"],
    "data": [
      [110.0, 26.0],
      [90.0, 28.0],
      [125.0, 24.0],
      [80.0, 29.0]
    ]
  }
}
```

### ml_class Collection

**Create config with normalization:**
```json
POST /api/v1/configs
{
  "name": "weather_model_normalized",
  "data": {
    "min_samples": 5,
    "normalize_features": true
  }
}
```

**Train with 3 features:**
```json
POST /api/v1/ml/$train
{
  "config_id": "{{config_id}}",
  "data": {
    "columns": ["rainfall", "mean_temperature", "humidity", "disease_cases"],
    "data": [
      [100.5, 25.3, 65.2, 12],
      [85.2, 27.1, 58.9, 8],
      [120.8, 24.5, 72.1, 15],
      [95.3, 26.2, 61.5, 10],
      [110.0, 26.0, 68.3, 13]
    ]
  }
}
```

### ml_shell Collection

**Train via external script:**
```json
POST /api/v1/ml/$train
{
  "config_id": "{{config_id}}",
  "data": {
    "columns": ["rainfall", "mean_temperature", "disease_cases"],
    "data": [
      [100.5, 25.3, 12],
      [85.2, 27.1, 8],
      [120.8, 24.5, 15],
      [95.3, 26.2, 10],
      [110.0, 26.0, 13]
    ]
  }
}
```

**Check server logs for script execution:**
```
executing_train_script command='python .../train_model.py ...'
train_script_completed stdout='Training completed'
```

## Environment Variables

To use collections across different environments (dev, staging, production), create Postman environments:

### Development Environment
```json
{
  "name": "Chapkit Dev",
  "values": [
    {
      "key": "baseUrl",
      "value": "http://127.0.0.1:8000",
      "enabled": true
    }
  ]
}
```

### Production Environment
```json
{
  "name": "Chapkit Production",
  "values": [
    {
      "key": "baseUrl",
      "value": "https://api.example.com",
      "enabled": true
    }
  ]
}
```

**To create environments:**
1. Click **Environments** (left sidebar)
2. Click **+** to create new environment
3. Add `baseUrl` variable with appropriate value
4. Select environment from dropdown (top-right)

## Pre-request Scripts

Collections include test scripts that automatically:
- Extract IDs from responses
- Store in collection variables
- Enable request chaining

**Example test script:**
```javascript
if (pm.response.code === 200 || pm.response.code === 202) {
    const response = pm.response.json();
    pm.collectionVariables.set('config_id', response.id);
}
```

## Troubleshooting

### Connection Refused
**Error:** `Error: connect ECONNREFUSED 127.0.0.1:8000`
**Solution:** Start the service with `fastapi dev examples/ml_basic.py`

### Variable Not Set
**Error:** `{{config_id}}` appears in URL instead of actual value
**Solution:** Run "Create Configuration" request first to auto-set the variable

### Job Still Pending
**Issue:** Prediction job never completes
**Solution:** Poll more frequently or check server logs for errors

### Missing Required Columns
**Error:** `"detail": "Missing required columns: humidity"`
**Solution:** Check which collection you're using - ml_class requires 3 features

### Script Not Found (ml_shell)
**Error:** `"Training script failed with exit code 1"`
**Solution:** Ensure `examples/scripts/train_model.py` exists and is executable

## Advanced Usage

### Testing Different Configurations

Create multiple configs and compare results:

1. Run `Create Config (With Normalization)`
2. Save `config_id` as `config_id_norm`
3. Run `Create Config (Without Normalization)`
4. Save `config_id` as `config_id_raw`
5. Train models with both configs
6. Compare prediction accuracy

### Batch Testing

Use Collection Runner with data files:

1. Create CSV with test data
2. Collection Runner → Select data file
3. Map CSV columns to request variables
4. Run collection with multiple iterations

### Monitoring and Logging

Enable Postman Console to see:
- Request/response details
- Variable updates
- Script execution logs

**To open:** View → Show Postman Console (or Cmd+Alt+C)

## Collection Maintenance

### Updating baseUrl

If service runs on different port:
1. Go to collection variables
2. Update `baseUrl` to `http://127.0.0.1:PORT`
3. Save collection

### Resetting Variables

To clear all auto-set IDs:
1. Click collection → Variables tab
2. Clear values for all auto-set variables
3. Save changes

## Next Steps

- **Read documentation:** See `ml_basic.md`, `ml_class.md`, `ml_shell.md` for detailed guides
- **Explore API:** Use Postman's documentation feature to generate API docs
- **Automate tests:** Convert collections to Newman scripts for CI/CD
- **Share collections:** Export and share with team members

## Related Files

- `ml_basic.md` - Detailed cURL guide for functional runner
- `ml_class.md` - Detailed cURL guide for class-based runner
- `ml_shell.md` - Detailed cURL guide for shell-based runner
- `quick_reference.md` - One-page cheat sheet
- `README.md` - General ML workflow overview

## Support

For issues or questions:
- Check example source code: `examples/ml_*.py`
- Review server logs for detailed error messages
- Consult CLAUDE.md for API reference
