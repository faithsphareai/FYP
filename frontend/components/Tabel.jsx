import React from 'react';
import { View, Text, TouchableOpacity, ScrollView } from 'react-native';

export default function Table ({ data, headers, loading }) {
  return (
    <View className="border rounded-lg border-gray-200 bg-white shadow-sm">
      {/* Table Header */}
      <View className="flex-row bg-green-50 px-4 py-3 border-b border-gray-200">
        {headers.map((header, index) => (
          <Text 
            key={header}
            className={`flex-1 text-sm font-semibold text-green-700 text-center ${
              index === 0 ? 'text-left' : 'text-center'
            }`}
          >
            {header}
          </Text>
        ))}
      </View>

      {/* Table Body */}
      <ScrollView>
        {loading ? (
          <View className="p-4">
            <Text className="text-center text-gray-500">Loading...</Text>
          </View>
        ) : data?.map((day, index) => (
          <TouchableOpacity 
            key={index}
            className={`
              flex-row px-4 py-3 border-b border-gray-100 
              ${index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}
              ${day.isToday ? 'bg-green-100' : ''}
            `}
          >
            <Text className="flex-1 text-base text-gray-900 text-center font-medium">
              {day.formattedDate}
            </Text>
            {['Fajr', 'Dhuhr', 'Asr', 'Maghrib', 'Isha'].map((prayer, prayerIndex) => (
              <Text 
                key={prayer}
                className={`flex-1 text-base text-gray-700 text-center ${
                  prayerIndex === 0 ? 'text-left' : 'text-center'
                }`}
              >
                {day.timings[prayer]}
              </Text>
            ))}
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
};