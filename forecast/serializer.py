from rest_framework import serializers
from .models import ForecastScenario


class ForecastScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = ForecastScenario
        fields = '__all__'


class GetScenarioById(serializers.Serializer):
    scenario_id = serializers.IntegerField()

    def validate(self, data):
        scenario_id = data.get('scenario_id')

        if not scenario_id:
            raise serializers.ValidationError({'error': 'scenario_id_not_provided'})

        return data


class FilterData(serializers.Serializer):
    filter_name = serializers.CharField(max_length=200)
    scenario_id = serializers.IntegerField()
    project_id = serializers.IntegerField()
    filter_value = serializers.CharField()

    def validate(self, data):
        filter_name = data.get('filter_name')
        scenario_id = data.get('scenario_id')
        project_id = data.get('project_id')
        filter_value = data.get('filter_value')

        filters = ['Family', 'Subcategory', 'Category', 'SKU', 'Salesman', 'Region', 'date', 'Client', 'client']

        if not filter_name:
            raise serializers.ValidationError({'error': 'filter_not_provided'})

        if filter_name not in filters:
            raise serializers.ValidationError({'error': 'invalid_filter'})

        if not scenario_id:
            raise serializers.ValidationError({'error': 'scenario_id_not_provided'})

        if not project_id:
            raise serializers.ValidationError({'error': 'project_id_not_provided'})

        if not filter_value:
            raise serializers.ValidationError({'error': 'filter_value_not_provided'})

        return data

