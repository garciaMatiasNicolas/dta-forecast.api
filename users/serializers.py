from rest_framework.serializers import ModelSerializer
from .models import User


class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        exclude = ('created_at',)

    def create(self, validated_data):
        user = User(**validated_data)
        user.set_password(validated_data['password'])
        user.save()
        return user

    def to_representation(self, instance):
        return {
            'id': instance.id,
            'email': instance.email,
            'first_name': instance.first_name,
            'last_name': instance.last_name
        }

