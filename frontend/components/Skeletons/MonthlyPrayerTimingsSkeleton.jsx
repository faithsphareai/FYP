import React from 'react';
import { View, ScrollView } from 'react-native';
import { createShimmerPlaceholder } from 'react-native-shimmer-placeholder';
import LinearGradient from 'expo-linear-gradient';

const ShimmerPlaceholder = createShimmerPlaceholder(LinearGradient);

export const MonthlyPrayerTimingsSkeleton = () => {
  // Shimmer configuration
  const shimmerConfig = {
    visible: false,
    shimmerColors: ['#f3f4f6', '#e5e7eb', '#f3f4f6'],
    duration: 1500,
    delay: 300,
    shimmerStyle: { borderRadius: 4 }
  };

  return (
    <ScrollView className="flex-1 bg-white" contentContainerStyle={{ paddingBottom: 20 }}>
      {/* Today's Prayers Section */}
      <View className="px-4 pt-6">
        <ShimmerPlaceholder
          {...shimmerConfig}
          style={{ width: 160, height: 20, borderRadius: 8, marginBottom: 16 }}
        />
        <View className="flex-row justify-between">
          {/* Current Prayer Card */}
          <View className="w-[48%] rounded-2xl p-4 bg-green-100">
            <View className="flex-row justify-between mb-3">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 112, height: 16 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 24, height: 24, borderRadius: 12 }}
              />
            </View>
            <View className="flex-row justify-between items-center">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 80, height: 20 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 64, height: 20 }}
              />
            </View>
          </View>

          {/* Upcoming Prayer Card */}
          <View className="w-[48%] rounded-2xl p-4 bg-gray-50">
            <View className="flex-row justify-between mb-3">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 112, height: 16 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 24, height: 24, borderRadius: 12 }}
              />
            </View>
            <View className="flex-row justify-between items-center">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 80, height: 20 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 64, height: 20 }}
              />
            </View>
          </View>
        </View>
      </View>

      {/* Sun Times Section */}
      <View className="px-4 pt-6">
        <ShimmerPlaceholder
          {...shimmerConfig}
          style={{ width: 120, height: 20, borderRadius: 8, marginBottom: 16 }}
        />
        <View className="flex-row justify-between">
          {/* Sunrise Card */}
          <View className="w-[48%] rounded-2xl p-4 bg-amber-50 flex flex-col">
            <ShimmerPlaceholder
              {...shimmerConfig}
              style={{ width: 32, height: 32, borderRadius: 16, marginBottom: 8, alignSelf: 'flex-end' }}
            />
            <View className="ml-3 flex flex-row items-center justify-between">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 60, height: 16 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 70, height: 16 }}
              />
            </View>
          </View>

          {/* Sunset Card */}
          <View className="w-[48%] rounded-2xl p-4 bg-indigo-50 flex flex-col">
            <ShimmerPlaceholder
              {...shimmerConfig}
              style={{ width: 32, height: 32, borderRadius: 16, marginBottom: 8, alignSelf: 'flex-end' }}
            />
            <View className="ml-3 flex flex-row items-center justify-between">
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 60, height: 16 }}
              />
              <ShimmerPlaceholder
                {...shimmerConfig}
                style={{ width: 70, height: 16 }}
              />
            </View>
          </View>
        </View>
      </View>

      {/* Monthly Table Section */}
      <View className="px-4 pt-6">
        <ShimmerPlaceholder
          {...shimmerConfig}
          style={{ width: 200, height: 20, borderRadius: 8, marginBottom: 16 }}
        />
        {/* Table Header */}
        <View className="flex-row bg-green-50 px-4 py-3 rounded-t-lg">
          {[...Array(6)].map((_, i) => (
            <ShimmerPlaceholder
              key={i}
              {...shimmerConfig}
              style={{ flex: 1, height: 16, marginHorizontal: 4 }}
            />
          ))}
        </View>
        {/* Table Rows */}
        {[...Array(15)].map((_, row) => (
          <View
            key={row}
            className={`flex-row px-4 py-3 ${row % 2 === 0 ? 'bg-gray-50' : 'bg-white'}`}
          >
            {[...Array(6)].map((__, col) => (
              <ShimmerPlaceholder
                key={col}
                {...shimmerConfig}
                style={{ flex: 1, height: 16, marginHorizontal: 4 }}
              />
            ))}
          </View>
        ))}
      </View>
    </ScrollView>
  );
};