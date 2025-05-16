import React from 'react';
import { Text, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import '../global.css';

export default function NimazCard({ bgColor = 'bg-white', namazName, title, time, icon, disabled }) {
  return (
    <View
      className={`p-4 rounded-xl shadow-sm ${bgColor} ${
        disabled ? 'opacity-80' : ''
      }`}
    >
      <View className="flex-row items-center justify-between mb-2">
        <Text className="text-xs text-gray-500 uppercase tracking-wide font-medium">{title}</Text>
        <Ionicons 
          name={icon} 
          size={20} 
          color={disabled ? '#6b7280' : '#15803d'} 
        />
      </View>

      <View className="flex-row justify-between items-center">
        <Text className={`text-lg font-semibold ${disabled ? 'text-gray-500' : 'text-gray-800'}`}>
          {namazName}
        </Text>
        <Text className={`text-lg font-medium ${disabled ? 'text-gray-400' : 'text-gray-600'}`}>
          {time}
        </Text>
      </View>
    </View>
  );
}