from rest_framework.routers import DefaultRouter
from .views.scenarios_views import ForecastScenarioViewSet

router_data_selector = DefaultRouter()

router_data_selector.register('', ForecastScenarioViewSet, basename='data_selectors')

urlpatterns = router_data_selector.urls