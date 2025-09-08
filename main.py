# simple demo of chapkit, will be moved out to its own repo sooon

from uuid import UUID

from chapkit.model import ChapModelConfig, ChapModelRunnerBase, ChapModelService
from chapkit.types import AssessedStatus, ChapServiceInfo
from chapkit.database import SqlAlchemyChapDatabase


class MyConfig(ChapModelConfig):
    x: int
    y: int


class MyRunner(ChapModelRunnerBase[MyConfig]): ...


info = ChapServiceInfo(
    author="Knut Rand",
    author_note=(
        "This model might need configuration of hyperparameters in order to work properly. "
        "When the model shows signs of overfitting, reduce 'state_dim' and/or increase "
        "'dropout' and 'weight_decay'."
    ),
    author_assessed_status=AssessedStatus.red,
    contact_email="knutdrand@gmail.com",
    description=(
        "This is a deep learning model template for CHAP. It is based on pytorch and can be used "
        "to train and predict using deep learning models. This typically need some configuration "
        "to fit the specifics of a dataset."
    ),
    display_name="Torch Deep Learning Model",
    organization="HISP Centre, University of Oslo",
    organization_logo_url="https://landportal.org/sites/default/files/2024-03/university_of_oslo_logo.png",
    citation_info=(
        'Climate Health Analytics Platform. 2025. "Torch Deep Learning Model". '
        "HISP Centre, University of Oslo. "
        "https://dhis2-chap.github.io/chap-core/external_models/overview_of_supported_models.html"
    ),
)

database = SqlAlchemyChapDatabase(file="target/chapkit.db", model_type=MyConfig)
runner = MyRunner(info, MyConfig, database)

default_config = MyConfig(id="06a0757d-3bea-4d74-b424-228fe7c1b2c2", name="default", x=10, y=20)
database.add_config(default_config)
database.add_config(MyConfig(id="aad4616c-e975-4cd8-b230-074eef580459", name="test", x=1, y=2))

database.add_artifact(
    id=UUID("06a0757d-3bea-4d74-b424-228fe7c1b2c2"),
    cfg=default_config,
    obj={"a": 1, "b": 2, "c": 3},
)

print(database.get_artifact(UUID("06a0757d-3bea-4d74-b424-228fe7c1b2c2")))

app = ChapModelService(
    runner=runner,
    database=database,
).create_fastapi()
