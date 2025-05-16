import React, { useState, useEffect } from "react";
import { View, Text, TouchableOpacity, ScrollView, Alert, Platform } from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { useLocalSearchParams, useRouter } from "expo-router";
import { Audio } from "expo-av";
import { Ionicons, MaterialIcons } from "@expo/vector-icons";
import * as Animatable from "react-native-animatable";

const AudioBar = ({ audio, onPlay, onPause, isPlaying }) => (
  <Animatable.View
    animation="fadeInUp"
    duration={600}
    className="flex-row items-center bg-green-50 rounded-lg p-3 mt-4 shadow-sm border border-green-100"
  >
    <Animatable.View
      animation={isPlaying ? "pulse" : undefined}
      iterationCount="infinite"
      duration={1000}
    >
      <MaterialIcons
        name="audiotrack"
        size={24}
        color={isPlaying ? "#15803d" : "#166534"}
        style={{ marginRight: 10 }}
      />
    </Animatable.View>
    <Text numberOfLines={1} className="flex-1 mr-2 text-lg font-medium text-green-800">
      {audio.name}
    </Text>
    {isPlaying ? (
      <Animatable.View animation="bounceIn" duration={300}>
        <TouchableOpacity
          onPress={onPause}
          className="bg-red-500 rounded-full w-10 h-10 justify-center items-center"
        >
          <Ionicons name="pause" size={24} color="white" />
        </TouchableOpacity>
      </Animatable.View>
    ) : (
      <Animatable.View animation="bounceIn" duration={300}>
        <TouchableOpacity
          onPress={onPlay}
          className="bg-green-600 rounded-full w-10 h-10 justify-center items-center"
        >
          <Ionicons name="play" size={24} color="white" />
        </TouchableOpacity>
      </Animatable.View>
    )}
  </Animatable.View>
);

export default function AuthenticationScreen() {
  const router = useRouter();
  const { title } = useLocalSearchParams();
  const [originalAudio, setOriginalAudio] = useState(null);
  const [userAudio, setUserAudio] = useState(null);
  const [sound, setSound] = useState();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState(null);
  const [similarityScore, setSimilarityScore] = useState(null);

  useEffect(() => {
    return sound
      ? () => {
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  const pickAudio = async (setAudio) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "audio/*",
        copyToCacheDirectory: true,
      });

      if (!result.canceled) {
        const file = {
          name: result.assets[0].name,
          uri: Platform.OS === "ios" 
            ? result.assets[0].uri.replace("file://", "") 
            : result.assets[0].uri,
        };
        setAudio(file);
        Alert.alert("Success", `Uploaded: ${file.name}`);
      }
    } catch (error) {
      Alert.alert("Error", "Failed to upload audio");
    }
  };

  const playSound = async (audio) => {
    if (audio) {
      try {
        if (sound) {
          await sound.unloadAsync();
        }

        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
          playsInSilentModeIOS: true,
          staysActiveInBackground: true,
          shouldDuckAndroid: true,
        });

        const { sound: newSound } = await Audio.Sound.createAsync(
          { uri: audio.uri },
          { shouldPlay: true }
        );
        setSound(newSound);
        setIsPlaying(true);
        setCurrentlyPlaying(audio);

        newSound.setOnPlaybackStatusUpdate((status) => {
          if (status.didJustFinish) {
            setIsPlaying(false);
            setCurrentlyPlaying(null);
          }
        });
      } catch (error) {
        Alert.alert("Error", "Failed to play audio");
      }
    }
  };

  const pauseSound = async () => {
    if (sound) {
      await sound.pauseAsync();
      setIsPlaying(false);
    }
  };

  const performTest = () => {
    if (!originalAudio || !userAudio) {
      Alert.alert("Error", "Please upload both original and user audio to proceed.");
      return;
    }
    const dummyScore = Math.floor(Math.random() * 100);
    setSimilarityScore(dummyScore);
  };

  return (
    <View className="flex-1 bg-white">
      <View className="pt-12 pb-4 bg-green-50 relative">
        <TouchableOpacity
          onPress={() => router.back()}
          className="absolute left-4 top-12"
        >
          <Ionicons name="arrow-back" size={24} color="#15803d" />
        </TouchableOpacity>
        <Text className="text-center text-xl font-bold text-green-800">
          {title || "Authentication"}
        </Text>
      </View>

      <ScrollView
        contentContainerStyle={{ padding: 20 }}
        showsVerticalScrollIndicator={false}
      >
        <Text className="text-lg font-bold  mt-4">
          Original Audio
        </Text>
        <TouchableOpacity
          onPress={() => pickAudio(setOriginalAudio)}
          className="border-2 border-gray-100 py-4 rounded-lg items-center mt-4  w-full"
        >
          <MaterialIcons name="upload-file" size={24} color="#16a34a" />
          <Text className=" text-lg mt-2">Upload</Text>
        </TouchableOpacity>
        {originalAudio && (
          <AudioBar
            audio={originalAudio}
            onPlay={() => playSound(originalAudio)}
            onPause={pauseSound}
            isPlaying={isPlaying && currentlyPlaying === originalAudio}
          />
        )}

        <Text className="text-lg font-bold  mt-6">
          User Audio
        </Text>
        <TouchableOpacity
          onPress={() => pickAudio(setUserAudio)}
          className="border-2 border-gray-100 py-4 rounded-lg items-center mt-4  w-full"
        >
          <MaterialIcons name="upload-file" size={24} color="#16a34a" />
          <Text className=" text-lg mt-2">Upload</Text>
        </TouchableOpacity>
        {userAudio && (
          <AudioBar
            audio={userAudio}
            onPlay={() => playSound(userAudio)}
            onPause={pauseSound}
            isPlaying={isPlaying && currentlyPlaying === userAudio}
          />
        )}

        <TouchableOpacity
          onPress={performTest}
          className="border-2 border-gray-100 py-4 rounded-lg items-center mt-6  w-full"
        >
          <Text className=" text-lg">
            <MaterialIcons name="search" size={20} color="#16a34a" /> Perform Test
          </Text>
        </TouchableOpacity>

        {similarityScore !== null && (
          <Animatable.View animation="fadeIn" duration={600} className="mt-6">
            <Text className="text-lg font-bold  text-center">
              Similarity Score: {similarityScore}%
            </Text>
          </Animatable.View>
        )}
      </ScrollView>
    </View>
  );
}