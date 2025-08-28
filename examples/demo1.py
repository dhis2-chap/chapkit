from chapkit import ChapConfig, ChapRunnerBase, ChapService, HealthResponse, HealthStatus, JsonChapStorage


class MyConfig(ChapConfig):
    x: int
    y: int


class MyRunner(ChapRunnerBase[MyConfig]):
    def on_health(self) -> HealthResponse:
        return HealthResponse(status=HealthStatus.up)


storage = JsonChapStorage("target/storage.json", MyConfig)
storage.add_config(MyConfig(id="06a0757d-3bea-4d74-b424-228fe7c1b2c2", name="default", x=10, y=20))
storage.add_config(MyConfig(id="aad4616c-e975-4cd8-b230-074eef580459", name="test", x=1, y=2))

service = ChapService(
    runner=MyRunner(),
    storage=storage,
    model_type=MyConfig,
)

app = service.create_fastapi()
