from rest_framework import serializers
from .file_model import FileRefModel, FileTypes


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileRefModel
        exclude = ('uploaded_at',)

    def to_representation(self, instance):
        return {
            'id': instance.id,
            'file_name': instance.file_name,
            'model_type': instance.model_type.model_type,
            'file': instance.file.url,
            'project': instance.project_id,
            'uploaded_at': instance.uploaded_at
        }


class FileModelType(serializers.ModelSerializer):
    class Meta:
        model = FileTypes
        fields = '__all__'


class ExcelToExportSerializer(serializers.Serializer):
    columns = serializers.ListSerializer(child=serializers.CharField())
    rows = serializers.ListSerializer(child=serializers.ListSerializer(child=serializers.CharField()))
    file_name = serializers.CharField(max_length=200)
    project_pk = serializers.IntegerField()

    def validate(self, data):
        columns = data.get('columns')
        rows = data.get('rows')
        file_name = data.get('file_name')
        project_pk = data.get('project_pk')

        if not columns:
            raise serializers.ValidationError({'error': 'columns_not_provided'})

        if not rows:
            raise serializers.ValidationError({'error': 'rows_not_provided'})

        if not file_name:
            raise serializers.ValidationError({'error': 'file_name_not_provided'})

        if not project_pk:
            raise serializers.ValidationError({'error': 'project_pk_not_provided'})

        return data