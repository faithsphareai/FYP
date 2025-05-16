import React, { useEffect, useState, useRef } from "react";
import {
  View,
  Text,
  Image,
  useWindowDimensions,
  TouchableOpacity,
  ActivityIndicator,
} from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import MapView, { Marker } from "react-native-maps";
import { useNearbyMosquesQuery } from "../../../hooks/useNearByMosqueQuery";
import getCurrentLocation from "../../../utils/getCurrentLocation";
import SearchButton from "../../../components/SearchButton";
import "../../../global.css";
import useNearbymosquesstore from "../../../zustand/NearByMosqueStore";

function GetNearbyMosques() {
  const router = useRouter();
  const [coords, setCoords] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const { width, height } = useWindowDimensions();
  const {
    data: mosques,
    isLoading,
    error,
    refetch,
    isRefetching,
  } = useNearbyMosquesQuery(coords);
  const { setMosques, mosquesData } = useNearbymosquesstore();
  const mapRef = useRef(null);
  const [initialRegion, setInitialRegion] = useState(null);
  // Calculate deltas (larger = more zoomed out)
  const ASPECT_RATIO = width / height;
  const LATITUDE_DELTA = 0.09; // tweak this for desired zoom
  const LONGITUDE_DELTA = LATITUDE_DELTA * ASPECT_RATIO;

  // Initial location fetch
  useEffect(() => {
    const fetchLocation = async () => {
      try {
        const location = await getCurrentLocation();
        if (location) setCoords(location);
      } catch (err) {
        console.error("Location Error:", err);
      }
    };
    fetchLocation();
  }, []);

  // Update Zustand store when mosques data changes
  useEffect(() => {
    if (Array.isArray(mosques)) {
      const mosquesWithId = mosques.map((m, idx) => ({
        ...m,
        id: `${idx}-${m.fullAddress || m.name}`,
      }));
      setMosques(mosquesWithId);
    }
  }, [mosques]);

  // Update useEffect for initial location fetch
  useEffect(() => {
    const fetchLocation = async () => {
      try {
        const location = await getCurrentLocation();
        if (location) {
          setCoords(location);
          setInitialRegion({
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            latitudeDelta: LATITUDE_DELTA,
            longitudeDelta: LONGITUDE_DELTA,
          });
        }
      } catch (err) {
        console.error("Location Error:", err);
      }
    };
    fetchLocation();
  }, []);

  // Add this function to center map on user location
  const handleCenterMap = async () => {
    try {
      const location = await getCurrentLocation();
      if (location) {
        setCoords(location);
        mapRef.current?.animateToRegion(
          {
            latitude: location.coords.latitude,
            longitude: location.coords.longitude,
            latitudeDelta: 0.02,
            longitudeDelta: 0.02 * (width / height),
          },
          1000
        );
      }
    } catch (error) {
      console.error("Error centering map:", error);
    }
  };

  const handleRefresh = async () => {
    try {
      setManualLoading(true);
      const location = await getCurrentLocation();
      setCoords(location);
      await refetch();
    } catch (err) {
      console.error("Refresh Error:", err);
    } finally {
      setManualLoading(false);
    }
  };

  const handleMosquePress = (mosque) => {
    router.push({
      pathname: "/mosque-modal",
      params: { id: mosque.id },
    });
  };

  // Combined loading states
  const showLoading = manualLoading || isLoading || isRefetching;

  if (showLoading) {
    return (
      <View className="flex-1 justify-center items-center bg-white">
        <ActivityIndicator size="large" color="#15803d" />
        <Text className="text-gray-600 mt-2">
          {manualLoading ? "Updating location..." : "Refreshing mosques..."}
        </Text>
      </View>
    );
  }

  if (error) {
    return (
      <View className="flex-1 justify-center items-center bg-white">
        <Text className="text-red-600 text-lg">Error: {error.message}</Text>
      </View>
    );
  }
  

  return (
    <SafeAreaView
      className="bg-white"
      edges={["top", "left", "right"]}
      style={{ flex: 1 }}
    >
      <LinearGradient colors={["#ffffff", "#f9fafb"]} style={{ flex: 1 }}>
        {/* Header with refresh button */}
        <View className="flex-row items-center px-4 py-3 border-b border-gray-200 bg-white">
          <TouchableOpacity
            onPress={() => router.back()}
            className="p-2 rounded-full bg-green-50"
          >
            <Ionicons name="arrow-back" size={24} color="#15803d" />
          </TouchableOpacity>

          <Text className="flex-1 text-center text-xl font-semibold text-green-700">
            Nearby Mosques
          </Text>

          <TouchableOpacity
            onPress={handleRefresh}
            className="p-2 rounded-full bg-green-50"
            disabled={showLoading}
          >
            {isRefetching ? (
              <ActivityIndicator size="small" color="#15803d" />
            ) : (
              <Ionicons name="refresh" size={24} color="#15803d" />
            )}
          </TouchableOpacity>
        </View>

        {/* Map Container */}
        <View style={{ flex: 1 }}>
          {initialRegion && (
            <MapView
              ref={mapRef}
              style={{ flex: 1 }}
              initialRegion={initialRegion}
              showsUserLocation={true}
              followsUserLocation={true}
            >
              {/* Markers remain the same */}
              {coords && (
                <Marker
                  coordinate={{
                    latitude: coords.coords.latitude,
                    longitude: coords.coords.longitude,
                  }}
                />
              )}

              {mosquesData.map((mosque) => (
                <Marker
                  key={mosque.id}
                  coordinate={{ latitude: mosque.lat, longitude: mosque.long }}
                  onPress={() => handleMosquePress(mosque)}
                >
                  <Image
                    source={require("../../../assets/mosqueLocationIcon.png")}
                    className="w-10 h-10"
                  />
                </Marker>
              ))}
            </MapView>
          )}
          {/* Add My Location Button */}
          {/* <TouchableOpacity
            className="absolute bottom-8 right-4 bg-white p-3 rounded-full shadow-lg"
            onPress={handleCenterMap}
          >
            <Ionicons name="locate" size={24} color="#15803d" />
          </TouchableOpacity> */}
        </View>
      </LinearGradient>
    </SafeAreaView>
  );
}

export default GetNearbyMosques;
