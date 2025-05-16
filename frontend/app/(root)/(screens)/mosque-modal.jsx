// MosqueModal.jsx
import { View, Text, Image, TouchableOpacity, Pressable } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { useLocalSearchParams, useRouter } from "expo-router";
import Animated, { FadeIn, SlideInDown } from "react-native-reanimated";
import useNearbymosquesstore from '../../../zustand/NearByMosqueStore';
import { AirbnbRating } from 'react-native-ratings';
import '../../../global.css';

function MosqueModal() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const { getMosqueById } = useNearbymosquesstore();
  
  const mosque = getMosqueById(params.id);

  if (!mosque) {
    return (
      <View className="flex-1 justify-center items-center bg-white">
        <Text className="text-red-500">Mosque not found</Text>
      </View>
    );
  }

  return (
    <Animated.View
      entering={FadeIn}
      style={{
        flex: 1,
        backgroundColor: "#00000040",
      }}
    >
      {/* Dismiss modal when pressing outside */}
      <Pressable 
        style={{ flex: 1 }} 
        onPress={() => router.back()}
      />

      {/* Content Container */}
      <Animated.View
        entering={SlideInDown}
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          backgroundColor: "white",
          borderTopLeftRadius: 20,
          borderTopRightRadius: 20,
          padding: 16,
          paddingBottom: 40,
          maxHeight: '60%'
        }}
      >
        <TouchableOpacity 
          style={{ position: 'absolute', top: 16, right: 16, zIndex:50 }}
          onPress={() => router.back()}
        >
          <Ionicons name="close" size={24} color="#262626" />
        </TouchableOpacity>

        {mosque.photos[0]?.src && (
          <Image
            source={{ uri: mosque.photos[0].src }}
            style={{ width: "100%", height: 120, borderRadius: 8, marginBottom: 16 }}
            resizeMode="cover"
          />
        )}

        <Text style={{ fontSize: 22, fontWeight: '700', color: '#1f2937', marginBottom: 8 }}>
          {mosque.name}
        </Text>

        {/* Rating Row */}
        <View className="flex-row items-center justify-between mb-4">
          <AirbnbRating
            count={5}
            defaultRating={mosque.rating || 0}
            size={20}
            showRating={false}
            isDisabled={true}
            selectedColor="#15803d"
            starContainerStyle={{ paddingVertical: 4 }}
          />
          
          {mosque.openNow && (
            <View className="flex-row items-center bg-green-100 px-3 py-1 rounded-full">
              <Ionicons name="checkmark-circle" size={16} color="#15803d" />
              <Text className="text-green-700 ml-2">Open Now</Text>
            </View>
          )}
        </View>

        {/* Address Section */}
        <View className="mb-4">
          <Text className="text-base text-gray-600 font-medium mb-1">Address</Text>
          <Text className="text-gray-500">{mosque.fullAddress}</Text>
        </View>

        {/* Coordinates */}
        <View className="flex-row justify-between">
          <View>
            <Text className="text-gray-600 font-medium">Latitude</Text>
            <Text className="text-gray-500">{mosque.lat?.toFixed(4)}</Text>
          </View>
          <View>
            <Text className="text-gray-600 font-medium">Longitude</Text>
            <Text className="text-gray-500">{mosque.long?.toFixed(4)}</Text>
          </View>
        </View>

      </Animated.View>
    </Animated.View>
  );
}

export default MosqueModal;