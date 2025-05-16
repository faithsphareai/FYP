import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import Tooltip from "react-native-walkthrough-tooltip";
import { usePrayerCalendarQuery } from "../../../hooks/usePrayerCalendarQuery";
import { parse, format } from "date-fns";
import Table from "../../../components/Tabel";
import NimazCard from "../../../components/NimazCard";
import { getCurrentAndNextPrayer } from "../.././../utils/getCurrentAndNextPrayer";
import "../../../global.css";
import { MonthlyPrayerTimingsSkeleton } from "../../../components/Skeletons/MonthlyPrayerTimingsSkeleton";
import getCurrentLocation from "../../../utils/getCurrentLocation";
import usePrayerSettingsStore from "../../../zustand/MonthlyPrayerStore";

export default function MonthlyPrayerTimings() {
  const router = useRouter();
  const [coords, setCoords] = useState(null);
  const [prayerData, setPrayerData] = useState([]);
  const [fetched, setFetched] = useState(false);
  const [isMounted, setIsMounted] = useState(true);
  const [showTip, setShowTip] = useState(false);
  const [todayTimes, setTodayTimes] = useState(null);

  const currentDate = new Date();
  const todayStr = format(currentDate, "dd-MM-yyyy");
  const { schoolShift } = usePrayerSettingsStore();
  const year = currentDate.getFullYear();
  const month = currentDate.getMonth() + 1;

  const { data, isLoading, error } = usePrayerCalendarQuery(
    coords,
    year,
    month,
    {
      enabled: !!coords && !fetched,
    },
    schoolShift === "shafi" ? 1 : 0
  );


  useEffect(() => {
    const fetchLocation = async () => {
      try {
        const location = await getCurrentLocation();
        if (location) {
          setCoords(location);
        }
      } catch (err) {
        console.error("Location Error:", err);
      }
    };

    fetchLocation();

    return () => setIsMounted(false);
  }, []);

  useEffect(() => {
    if (data && !fetched && isMounted) {
      const processedData = data.map((day) => ({
        ...day,
        formattedDate: format(
          parse(day.date.gregorian.date, "dd-MM-yyyy", new Date()),
          "dd MMM"
        ),
        timings: Object.fromEntries(
          Object.entries(day.timings).map(([key, value]) => [
            key,
            value.split(" ")[0],
          ])
        ),
        isToday: day.date.gregorian.date === todayStr,
      }));

      setPrayerData(processedData);
      const todayEntry = processedData.find((d) => d.isToday);
      if (todayEntry) {
        setTodayTimes(todayEntry.timings);
      }
      setFetched(true);
    }
  }, [data, fetched, isMounted]);

  const { currentPrayer, nextPrayer } = todayTimes
    ? getCurrentAndNextPrayer(todayTimes)
    : {};

  return (
    <SafeAreaView className="flex-1 bg-white" edges={["top", "left", "right"]}>
      <LinearGradient colors={["#ffffff", "#f9fafb"]} className="flex-1">
        <View className="flex-row items-center px-4 py-3 border-b border-gray-200 bg-white">
          <TouchableOpacity
            onPress={() => router.back()}
            className="p-2 rounded-full bg-green-50"
          >
            <Ionicons name="arrow-back" size={24} color="#15803d" />
          </TouchableOpacity>
          <Text className="flex-1 text-center text-xl font-semibold text-green-700">
            Monthly Prayer Timings
          </Text>
          {/* Info Button */}
          <Tooltip
            isVisible={showTip}
            topAdjustment={-33.5}
            placement="bottom"
            onClose={() => setShowTip(false)}
            
            tooltipStyle={{
              backgroundColor: "#ffffff",
              borderRadius: 8,
              padding: 10,
              shadowColor: "#000",
              shadowOffset: { width: 0, height: 2 },
              shadowOpacity: 0.25,
              shadowRadius: 4,
              elevation: 5,
            }}
            content={
              <Text className="text-sm text-gray-700">
                Below prayer timing was in{" "}
                <Text className="font-bold text-[#15803d]">
                  {schoolShift.charAt(0).toUpperCase() + schoolShift.slice(1)}
                </Text>
                 to change this go to{" "}
                <Text
                  className="font-bold underline text-[#15803d]"
                  onPress={() => {
                    setShowTip(false);
                    router.push("/profile");
                  }}
                >
                  Settings
                </Text>
                .
              </Text>
            }
          >
            <TouchableOpacity
              onPress={() => setShowTip(true)}
              className="p-2 rounded-full bg-green-50"
            >
              <Ionicons
                name="information-circle-outline"
                size={24}
                color="#15803d"
              />
            </TouchableOpacity>
          </Tooltip>
        </View>

        {isLoading && !fetched ? (
          <MonthlyPrayerTimingsSkeleton />
        ) : error ? (
          <View className="flex-1 justify-center items-center">
            <Text className="text-red-600 text-lg">Failed to load data</Text>
          </View>
        ) : (
          <ScrollView className="flex-1">
            {/* Current & Upcoming Prayer Section */}
            <View className="px-4 pt-4">
              <Text className="text-lg font-semibold text-gray-800 mb-3">
                Today's Prayers
              </Text>
              <View className="flex-row justify-between">
                <View className="w-[48%]">
                  <NimazCard
                    bgColor="bg-green-100"
                    namazName={currentPrayer?.name}
                    title="Current Prayer"
                    time={currentPrayer?.time}
                    icon="checkmark-circle"
                  />
                </View>
                <View className="w-[48%]">
                  <NimazCard
                    bgColor="bg-gray-50"
                    namazName={nextPrayer?.name}
                    title="Upcoming Prayer"
                    time={nextPrayer?.time}
                    icon="time"
                    disabled
                  />
                </View>
              </View>
            </View>

            {/* Sunrise & Sunset Section */}
            <View className="px-4 pt-6">
              <Text className="text-lg font-semibold text-gray-800 mb-3">
                Sun Times
              </Text>
              <View className="flex-row justify-between">
                <View className="w-[48%]">
                  <NimazCard
                    bgColor="bg-amber-50"
                    namazName="Sunrise"
                    time={todayTimes?.Sunrise}
                    icon="sunny"
                  />
                </View>
                <View className="w-[48%]">
                  <NimazCard
                    bgColor="bg-indigo-50"
                    namazName="Sunset"
                    time={todayTimes?.Sunset}
                    icon="moon"
                  />
                </View>
              </View>
            </View>

            {/* Monthly Timings Table */}
            <View className="p-4 pt-6">
              <Text className="text-lg font-semibold text-gray-800 mb-3">
                Monthly Schedule
              </Text>
              <Table
                headers={["Date", "Fajr", "Dhuhr", "Asr", "Maghrib", "Isha"]}
                data={prayerData}
                loading={isLoading}
              />
            </View>
          </ScrollView>
        )}
      </LinearGradient>
    </SafeAreaView>
  );
}
