from django.conf import settings
from sqlalchemy.engine import create_engine

database_name = settings.DATABASES['default']['NAME']
database_url = 'sqlite:///{}'.format(database_name)
engine = create_engine(database_url, echo=False)