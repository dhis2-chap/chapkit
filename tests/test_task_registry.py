"""Tests for TaskRegistry functionality."""

from collections.abc import Generator

import pytest

from chapkit.task import TaskRegistry


@pytest.fixture(autouse=True)
def clear_registry() -> Generator[None, None, None]:
    """Clear registry before and after each test."""
    TaskRegistry.clear()
    yield
    TaskRegistry.clear()


def test_register_decorator() -> None:
    """Test registering a function using the decorator."""

    @TaskRegistry.register("test_func")
    def test_func() -> str:
        return "test"

    assert "test_func" in TaskRegistry.list_all()
    func = TaskRegistry.get("test_func")
    assert func() == "test"


def test_register_function_imperative() -> None:
    """Test registering a function imperatively."""

    def my_func() -> str:
        return "my result"

    TaskRegistry.register_function("my_func", my_func)

    assert "my_func" in TaskRegistry.list_all()
    func = TaskRegistry.get("my_func")
    assert func() == "my result"


def test_register_async_function() -> None:
    """Test registering an async function."""

    @TaskRegistry.register("async_func")
    async def async_func() -> str:
        return "async result"

    assert "async_func" in TaskRegistry.list_all()
    func = TaskRegistry.get("async_func")
    assert callable(func)


def test_duplicate_registration_decorator() -> None:
    """Test that duplicate registration raises ValueError."""

    @TaskRegistry.register("dup_func")
    def func1() -> str:
        return "first"

    with pytest.raises(ValueError, match="Task 'dup_func' already registered"):

        @TaskRegistry.register("dup_func")
        def func2() -> str:
            return "second"


def test_duplicate_registration_imperative() -> None:
    """Test that duplicate imperative registration raises ValueError."""

    def func1() -> str:
        return "first"

    def func2() -> str:
        return "second"

    TaskRegistry.register_function("dup_func", func1)

    with pytest.raises(ValueError, match="Task 'dup_func' already registered"):
        TaskRegistry.register_function("dup_func", func2)


def test_get_missing_function() -> None:
    """Test that getting a missing function raises KeyError."""
    with pytest.raises(KeyError, match="Task 'missing' not found in registry"):
        TaskRegistry.get("missing")


def test_list_all_empty() -> None:
    """Test listing all tasks when registry is empty."""
    assert TaskRegistry.list_all() == []


def test_list_all_multiple() -> None:
    """Test listing all registered tasks."""

    @TaskRegistry.register("func_a")
    def func_a() -> None:
        pass

    @TaskRegistry.register("func_c")
    def func_c() -> None:
        pass

    @TaskRegistry.register("func_b")
    def func_b() -> None:
        pass

    tasks = TaskRegistry.list_all()
    assert tasks == ["func_a", "func_b", "func_c"]  # Should be sorted


def test_clear() -> None:
    """Test clearing the registry."""

    @TaskRegistry.register("func1")
    def func1() -> None:
        pass

    @TaskRegistry.register("func2")
    def func2() -> None:
        pass

    assert len(TaskRegistry.list_all()) == 2

    TaskRegistry.clear()

    assert TaskRegistry.list_all() == []


def test_register_with_parameters() -> None:
    """Test registering function that accepts parameters."""

    @TaskRegistry.register("add_numbers")
    def add_numbers(a: int, b: int) -> int:
        return a + b

    func = TaskRegistry.get("add_numbers")
    assert func(5, 3) == 8
    assert func(a=10, b=20) == 30


def test_register_with_default_parameters() -> None:
    """Test registering function with default parameters."""

    @TaskRegistry.register("greet")
    def greet(name: str, greeting: str = "Hello") -> str:
        return f"{greeting}, {name}!"

    func = TaskRegistry.get("greet")
    assert func("World") == "Hello, World!"
    assert func("World", greeting="Hi") == "Hi, World!"


def test_register_with_tags() -> None:
    """Test registering tasks with tags."""

    @TaskRegistry.register("task1", tags=["tag1", "tag2"])
    def task1() -> str:
        return "result"

    assert TaskRegistry.has("task1")
    tags = TaskRegistry.get_tags("task1")
    assert tags == ["tag1", "tag2"]


def test_register_without_tags() -> None:
    """Test registering task without tags defaults to empty list."""

    @TaskRegistry.register("task_no_tags")
    def task_no_tags() -> str:
        return "result"

    tags = TaskRegistry.get_tags("task_no_tags")
    assert tags == []


def test_get_tags_missing() -> None:
    """Test getting tags for non-existent task raises KeyError."""
    with pytest.raises(KeyError, match="Task 'missing' not found in registry"):
        TaskRegistry.get_tags("missing")


def test_list_by_tags_empty() -> None:
    """Test filtering by tags when no tasks match."""

    @TaskRegistry.register("task1", tags=["tag1"])
    def task1() -> None:
        pass

    result = TaskRegistry.list_by_tags(["tag2"])
    assert result == []


def test_list_by_tags_single() -> None:
    """Test filtering by single tag."""

    @TaskRegistry.register("task1", tags=["data", "etl"])
    def task1() -> None:
        pass

    @TaskRegistry.register("task2", tags=["data"])
    def task2() -> None:
        pass

    result = TaskRegistry.list_by_tags(["data"])
    assert result == ["task1", "task2"]


def test_list_by_tags_multiple_and_logic() -> None:
    """Test filtering by multiple tags requires ALL tags."""

    @TaskRegistry.register("task1", tags=["data", "etl"])
    def task1() -> None:
        pass

    @TaskRegistry.register("task2", tags=["data"])
    def task2() -> None:
        pass

    @TaskRegistry.register("task3", tags=["data", "etl", "extract"])
    def task3() -> None:
        pass

    # Requires ALL tags
    result = TaskRegistry.list_by_tags(["data", "etl"])
    assert result == ["task1", "task3"]


def test_list_by_tags_no_tags() -> None:
    """Test filtering with empty tag list returns all tasks."""

    @TaskRegistry.register("task1", tags=["tag1"])
    def task1() -> None:
        pass

    @TaskRegistry.register("task2", tags=["tag2"])
    def task2() -> None:
        pass

    result = TaskRegistry.list_by_tags([])
    assert result == ["task1", "task2"]


def test_get_info() -> None:
    """Test getting task metadata."""

    @TaskRegistry.register("info_task", tags=["demo"])
    def info_task(a: int, b: str = "default") -> dict:
        """Example task docstring."""
        return {}

    info = TaskRegistry.get_info("info_task")
    assert info.name == "info_task"
    assert info.docstring == "Example task docstring."
    assert info.tags == ["demo"]
    assert len(info.parameters) == 2

    # Check parameter info
    param_a = next(p for p in info.parameters if p.name == "a")
    assert param_a.required is True
    assert param_a.annotation == "<class 'int'>"

    param_b = next(p for p in info.parameters if p.name == "b")
    assert param_b.required is False
    assert param_b.default == "default"


def test_get_info_missing() -> None:
    """Test getting info for non-existent task raises KeyError."""
    with pytest.raises(KeyError, match="Task 'missing' not found in registry"):
        TaskRegistry.get_info("missing")


def test_list_all_info() -> None:
    """Test listing all task info."""

    @TaskRegistry.register("task1", tags=["tag1"])
    def task1() -> None:
        """Task 1 docstring."""
        pass

    @TaskRegistry.register("task2", tags=["tag2"])
    def task2() -> None:
        """Task 2 docstring."""
        pass

    all_info = TaskRegistry.list_all_info()
    assert len(all_info) == 2
    assert all_info[0].name == "task1"
    assert all_info[1].name == "task2"


def test_has() -> None:
    """Test checking if task is registered."""

    @TaskRegistry.register("exists")
    def exists() -> None:
        pass

    assert TaskRegistry.has("exists") is True
    assert TaskRegistry.has("not_exists") is False


def test_url_safe_name_validation() -> None:
    """Test that task names must be URL-safe."""

    # Valid names
    @TaskRegistry.register("valid_name")
    def valid1() -> None:
        pass

    @TaskRegistry.register("valid-name")
    def valid2() -> None:
        pass

    @TaskRegistry.register("ValidName123")
    def valid3() -> None:
        pass

    # Invalid names
    with pytest.raises(ValueError, match="must be URL-safe"):

        @TaskRegistry.register("invalid name")
        def invalid1() -> None:
            pass

    with pytest.raises(ValueError, match="must be URL-safe"):

        @TaskRegistry.register("invalid/name")
        def invalid2() -> None:
            pass

    with pytest.raises(ValueError, match="must be URL-safe"):

        @TaskRegistry.register("invalid.name")
        def invalid3() -> None:
            pass


def test_register_function_url_safe_validation() -> None:
    """Test URL-safe validation for imperative registration."""

    def valid_func() -> None:
        pass

    # Valid name
    TaskRegistry.register_function("valid-name", valid_func)
    assert TaskRegistry.has("valid-name")

    # Invalid name
    with pytest.raises(ValueError, match="must be URL-safe"):
        TaskRegistry.register_function("invalid name", valid_func)


def test_get_info_with_varargs() -> None:
    """Test get_info with *args and **kwargs parameters."""

    @TaskRegistry.register("task_with_varargs", tags=["test"])
    def task_with_varargs(a: int, *args, **kwargs) -> dict:
        """Task with varargs."""
        return {}

    info = TaskRegistry.get_info("task_with_varargs")
    assert info.name == "task_with_varargs"
    # *args and **kwargs should be excluded from parameters
    assert len(info.parameters) == 1
    assert info.parameters[0].name == "a"
