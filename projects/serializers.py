from rest_framework.serializers import ModelSerializer
from .models import ProjectsModel


class ProjectSerializer(ModelSerializer):
    class Meta:
        model = ProjectsModel
        exclude = ('created_at', 'status')

    def to_representation(self, instance):
        return {
            'id': instance.id,
            'project_name': instance.project_name,
            'created_at': instance.created_at,
            'status': instance.status,
            'user_owner': f'{instance.user_owner.first_name} {instance.user_owner.last_name}'
        }