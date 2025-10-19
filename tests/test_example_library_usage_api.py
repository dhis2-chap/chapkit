"""Tests for library_usage_api example using TestClient.

This example demonstrates using chapkit as a library with custom models.
Tests use FastAPI's TestClient instead of running a separate server.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from examples.library_usage_api import app


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Create FastAPI TestClient for testing with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_landing_page(client: TestClient) -> None:
    """Test landing page returns HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_health_endpoint(client: TestClient) -> None:
    """Test health check returns healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_list_configs(client: TestClient) -> None:
    """Test listing configs using chapkit's Config model."""
    response = client.get("/api/v1/configs")
    assert response.status_code == 200
    data = response.json()

    # Should have at least the seeded production config
    assert isinstance(data, list)
    assert len(data) >= 1

    # Find production config
    production = next((c for c in data if c["name"] == "production"), None)
    assert production is not None
    assert "data" in production
    assert "max_users" in production["data"]
    assert "registration_enabled" in production["data"]
    assert "default_theme" in production["data"]


def test_create_config(client: TestClient) -> None:
    """Test creating a config with ApiConfig schema."""
    new_config = {"name": "staging", "data": {"max_users": 500, "registration_enabled": True, "default_theme": "light"}}

    response = client.post("/api/v1/configs", json=new_config)
    assert response.status_code == 201
    created = response.json()

    assert created["name"] == "staging"
    assert created["data"]["max_users"] == 500
    assert created["data"]["default_theme"] == "light"


def test_list_users(client: TestClient) -> None:
    """Test listing users using custom User model."""
    response = client.get("/api/v1/users")
    assert response.status_code == 200
    data = response.json()

    # Should have at least the seeded admin user
    assert isinstance(data, list)
    assert len(data) >= 1

    # Find admin user
    admin = next((u for u in data if u["username"] == "admin"), None)
    assert admin is not None
    assert admin["email"] == "admin@example.com"
    assert admin["full_name"] == "Administrator"
    assert "preferences" in admin
    assert admin["preferences"]["theme"] == "dark"


def test_get_user_by_id(client: TestClient) -> None:
    """Test retrieving user by ID."""
    # Get list to find admin user ID
    list_response = client.get("/api/v1/users")
    users = list_response.json()
    admin = next((u for u in users if u["username"] == "admin"), None)
    assert admin is not None
    user_id = admin["id"]

    # Get user by ID
    response = client.get(f"/api/v1/users/{user_id}")
    assert response.status_code == 200
    user = response.json()

    assert user["id"] == user_id
    assert user["username"] == "admin"
    assert user["email"] == "admin@example.com"


def test_get_user_not_found(client: TestClient) -> None:
    """Test retrieving non-existent user returns 404."""
    response = client.get("/api/v1/users/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_create_user(client: TestClient) -> None:
    """Test creating a user with custom User model."""
    new_user = {
        "username": "newuser",
        "email": "newuser@example.com",
        "full_name": "New User",
        "preferences": {"theme": "light", "notifications": False, "language": "en"},
    }

    response = client.post("/api/v1/users", json=new_user)
    assert response.status_code == 201
    created = response.json()

    assert "id" in created
    assert created["username"] == "newuser"
    assert created["email"] == "newuser@example.com"
    assert created["full_name"] == "New User"
    assert created["preferences"]["theme"] == "light"
    assert created["preferences"]["notifications"] is False


def test_create_user_duplicate_username(client: TestClient) -> None:
    """Test creating user with duplicate username fails."""
    duplicate_user = {
        "username": "admin",  # Already exists
        "email": "different@example.com",
        "full_name": "Different User",
    }

    response = client.post("/api/v1/users", json=duplicate_user)
    # Should fail due to unique constraint on username
    assert response.status_code in [400, 409, 500]


def test_create_user_duplicate_email(client: TestClient) -> None:
    """Test creating user with duplicate email fails."""
    duplicate_user = {
        "username": "different",
        "email": "admin@example.com",  # Already exists
        "full_name": "Different User",
    }

    response = client.post("/api/v1/users", json=duplicate_user)
    # Should fail due to unique constraint on email
    assert response.status_code in [400, 409, 500]


def test_create_user_with_minimal_fields(client: TestClient) -> None:
    """Test creating user with only required fields."""
    minimal_user = {
        "username": "minimal",
        "email": "minimal@example.com",
        # full_name and preferences are optional
    }

    response = client.post("/api/v1/users", json=minimal_user)
    assert response.status_code == 201
    created = response.json()

    assert created["username"] == "minimal"
    assert created["email"] == "minimal@example.com"
    assert created["full_name"] is None
    assert created["preferences"] == {}


def test_create_user_invalid_email(client: TestClient) -> None:
    """Test creating user with invalid email format fails."""
    invalid_user = {
        "username": "invalidtest",
        "email": "not-an-email",
        "full_name": "Invalid Test",
    }

    response = client.post("/api/v1/users", json=invalid_user)
    assert response.status_code == 422
    data = response.json()
    assert "email" in str(data).lower()


def test_update_user(client: TestClient) -> None:
    """Test updating a user."""
    # Create a user first
    new_user = {
        "username": "updatetest",
        "email": "updatetest@example.com",
        "full_name": "Update Test",
        "preferences": {"theme": "dark"},
    }
    create_response = client.post("/api/v1/users", json=new_user)
    created = create_response.json()
    user_id = created["id"]

    # Update the user
    updated_user = {
        "id": user_id,
        "username": "updatetest",
        "email": "updatetest@example.com",
        "full_name": "Updated Name",  # Changed
        "preferences": {"theme": "light", "notifications": True},  # Changed
    }

    response = client.put(f"/api/v1/users/{user_id}", json=updated_user)
    assert response.status_code == 200
    updated = response.json()

    assert updated["id"] == user_id
    assert updated["full_name"] == "Updated Name"
    assert updated["preferences"]["theme"] == "light"
    assert updated["preferences"]["notifications"] is True


def test_delete_user(client: TestClient) -> None:
    """Test deleting a user."""
    # Create a user first
    new_user = {"username": "deletetest", "email": "deletetest@example.com", "full_name": "Delete Test"}
    create_response = client.post("/api/v1/users", json=new_user)
    created = create_response.json()
    user_id = created["id"]

    # Delete the user
    response = client.delete(f"/api/v1/users/{user_id}")
    assert response.status_code == 204

    # Verify user is deleted
    get_response = client.get(f"/api/v1/users/{user_id}")
    assert get_response.status_code == 404


def test_delete_user_not_found(client: TestClient) -> None:
    """Test deleting non-existent user returns 404."""
    response = client.delete("/api/v1/users/01K72P5N5KCRM6MD3BRE4P0999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_user_preferences_json_field(client: TestClient) -> None:
    """Test that user preferences are stored as JSON and can be queried."""
    # Create a user with complex preferences
    new_user = {
        "username": "jsontest",
        "email": "jsontest@example.com",
        "preferences": {
            "theme": "auto",
            "notifications": True,
            "language": "es",
            "timezone": "Europe/Madrid",
            "custom_field": {"nested": "value"},
        },
    }

    response = client.post("/api/v1/users", json=new_user)
    assert response.status_code == 201
    created = response.json()
    user_id = created["id"]

    # Retrieve and verify preferences
    get_response = client.get(f"/api/v1/users/{user_id}")
    user = get_response.json()

    prefs = user["preferences"]
    assert prefs["theme"] == "auto"
    assert prefs["notifications"] is True
    assert prefs["language"] == "es"
    assert prefs["timezone"] == "Europe/Madrid"
    assert prefs["custom_field"]["nested"] == "value"


def test_list_users_with_pagination(client: TestClient) -> None:
    """Test listing users with pagination."""
    # Create a few more users to test pagination
    for i in range(3):
        client.post("/api/v1/users", json={"username": f"pagetest{i}", "email": f"pagetest{i}@example.com"})

    # Get paginated list
    response = client.get("/api/v1/users", params={"page": 1, "size": 2})
    assert response.status_code == 200
    data = response.json()

    # Should return paginated response
    assert "items" in data
    assert "total" in data
    assert "page" in data
    assert "size" in data
    assert "pages" in data

    assert len(data["items"]) <= 2
    assert data["page"] == 1
    assert data["size"] == 2


def test_config_and_user_coexist(client: TestClient) -> None:
    """Test that Config and User endpoints coexist properly."""
    # Both endpoints should work
    config_response = client.get("/api/v1/configs")
    assert config_response.status_code == 200

    user_response = client.get("/api/v1/users")
    assert user_response.status_code == 200

    # Both should return valid data
    configs = config_response.json()
    users = user_response.json()

    assert isinstance(configs, list)
    assert isinstance(users, list)
    assert len(configs) >= 1
    assert len(users) >= 1
