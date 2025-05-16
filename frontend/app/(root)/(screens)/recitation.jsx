import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  Platform,
  Pressable,
  StyleSheet,
  ActivityIndicator,
  Image,
} from "react-native";
import * as DocumentPicker from "expo-document-picker";
import { Audio } from "expo-av";
import { MaterialIcons, Ionicons, FontAwesome5 } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import * as Animatable from "react-native-animatable";
import { useNavigation } from "@react-navigation/native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useCompareDTW } from "../../../hooks/useCompareDTW";
import { LinearGradient } from "expo-linear-gradient";

const AudioBar = ({
  audio,
  onPlay,
  onPause,
  isPlaying,
  setAudio,
  setIslUpload,
}) => (
  <Animatable.View
    animation="fadeInUp"
    duration={600}
    style={styles.audioBarContainer}
  >
    <LinearGradient
      colors={["#f0fdf4", "#dcfce7"]}
      style={styles.audioBarGradient}
    >
      <TouchableOpacity
        style={styles.removeButton}
        onPress={() => {
          setAudio(null);
          setIslUpload(false);
          onPause();
        }}
      >
        <Text style={styles.removeButtonText}>×</Text>
      </TouchableOpacity>
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
      <Text numberOfLines={1} style={styles.audioTitle}>
        {audio.name || "Unnamed Audio"}
      </Text>
      {isPlaying ? (
        <Animatable.View animation="bounceIn" duration={300}>
          <TouchableOpacity onPress={onPause} style={styles.pauseButton}>
            <Ionicons name="pause" size={24} color="white" />
          </TouchableOpacity>
        </Animatable.View>
      ) : (
        <Animatable.View animation="bounceIn" duration={300}>
          <TouchableOpacity onPress={onPlay} style={styles.playButton}>
            <Ionicons name="play" size={24} color="white" />
          </TouchableOpacity>
        </Animatable.View>
      )}
    </LinearGradient>
  </Animatable.View>
);

export default function Recitation() {
  const [originalAudio, setOriginalAudio] = useState(null);
  const [userAudio, setUserAudio] = useState(null);
  const [sound, setSound] = useState();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentlyPlaying, setCurrentlyPlaying] = useState(null);
  const [isOriginalUpload, setIsOriginalUpload] = useState(false);
  const [isUserUpload, setIsUserUpload] = useState(false);
  const router = useRouter();

  const compareDTWMutation = useCompareDTW();

  const [originalRecording, setOriginalRecording] = useState();
  const [userRecording, setUserRecording] = useState();
  const [isOriginalRecording, setIsOriginalRecording] = useState(false);
  const [isUserRecording, setIsUserRecording] = useState(false);

  const navigation = useNavigation();

  useEffect(() => {
    return sound
      ? () => {
          sound.unloadAsync();
        }
      : undefined;
  }, [sound]);

  const pickAudio = async (setter) => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "audio/*",
        copyToCacheDirectory: true,
      });

      if (result.canceled == true) {
        return false;
      }

      if (!result.canceled) {
        // Format the file object correctly
        const file = {
          name: result.assets[0].name,
          uri:
            Platform.OS === "ios"
              ? result.assets[0].uri.replace("file://", "")
              : result.assets[0].uri,
        };
        setter(file);
        Alert.alert("Success", `Uploaded: ${file.name}`);
        return true;
      }
    } catch (error) {
      console.error("Audio Upload Error:", error);
      Alert.alert("Error", "Failed to upload audio.");
      return false;
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
          { shouldPlay: true },
          (status) => {
            if (status.error) {
              console.error(`Audio playback error: ${status.error}`);
              Alert.alert("Playback Error", "Failed to play audio file");
            }
          }
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
        console.error("Play Sound Error:", error);
        Alert.alert(
          "Error",
          "Failed to play audio. Please try uploading again."
        );
      }
    }
  };

  const pauseSound = async () => {
    if (sound) {
      await sound.pauseAsync();
      setIsPlaying(false);
    }
  };

  const similaritySearch = async () => {
    if (!originalAudio || !userAudio) {
      Alert.alert(
        "Missing Audio",
        "Please upload or record both original and user audio files."
      );
      return;
    }

    try {
      const result = await compareDTWMutation.mutateAsync({
        originalAudio: {
          ...originalAudio,
          type: originalAudio.name.endsWith(".mp3") ? "audio/mp3" : "audio/ogg",
        },
        userAudio: {
          ...userAudio,
          type: userAudio.name.endsWith(".mp3") ? "audio/mp3" : "audio/ogg",
        },
      });

      // Check if result is valid and has the expected properties
      if (result && typeof result.similarity_score === "number") {
        Alert.alert(
          "Similarity Results",
          `Score: ${result.similarity_score.toFixed(2)}%\n\nInterpretation: ${
            result.interpretation || "No interpretation available"
          }`
        );
      } else {
        throw new Error("Invalid response format from comparison service");
      }
    } catch (error) {
      console.error("Full error details:", error);

      Alert.alert(
        "Error",
        "An error occurred while comparing the audio files. Please try again later."
      );
    }
  };

  async function startRecording(isOriginal) {
    try {
      const audioPermission = await Audio.requestPermissionsAsync();
      if (!audioPermission.granted) {
        Alert.alert("Permission Required", "Please grant microphone access");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: true,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
        (status) => console.log("Recording status:", status),
        100
      );

      if (isOriginal) {
        setOriginalRecording(recording);
        setIsOriginalRecording(true);
      } else {
        setUserRecording(recording);
        setIsUserRecording(true);
      }
    } catch (err) {
      console.error("Recording error:", err);
      Alert.alert(
        "Error",
        "Failed to start recording. Please check microphone permissions."
      );
    }
  }

  async function stopRecording(isOriginal) {
    try {
      const currentRecording = isOriginal ? originalRecording : userRecording;
      if (!currentRecording) return;

      if (isOriginal) {
        setIsOriginalRecording(false);
      } else {
        setIsUserRecording(false);
      }

      await currentRecording.stopAndUnloadAsync();
      const uri = currentRecording.getURI();
      const fileName = `recording-${Date.now()}.m4a`;

      const audioFile = {
        name: fileName,
        uri: Platform.OS === "ios" ? uri.replace("file://", "") : uri,
      };

      if (isOriginal) {
        setOriginalAudio(audioFile);
        setIsOriginalUpload(true);
        setOriginalRecording(undefined);
      } else {
        setUserAudio(audioFile);
        setIsUserUpload(true);
        setUserRecording(undefined);
      }
    } catch (err) {
      Alert.alert("Error", "Failed to stop recording");
    }
  }

  const recordAudio = async (isOriginal) => {
    if (isOriginal) {
      if (isOriginalRecording) {
        await stopRecording(true);
      } else {
        await startRecording(true);
      }
    } else {
      if (isUserRecording) {
        await stopRecording(false);
      } else {
        await startRecording(false);
      }
    }
  };

  const handleBack = () => {
    router.back();
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <LinearGradient colors={["#ffffff", "#f9fafb"]} style={styles.container}>
        <View style={styles.header}>
          <TouchableOpacity onPress={handleBack} style={styles.backButton}>
            <Ionicons name="arrow-back" size={24} color="#15803d" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Recitation Comparison</Text>
          <View style={styles.headerSpacer} />
        </View>

        <View style={styles.introContainer}>
          <Text style={styles.introText}>
            Compare your recitation with the original to improve your
            pronunciation and rhythm.
          </Text>
        </View>

        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.sectionContainer}>
            <View style={styles.sectionHeaderRow}>
              <FontAwesome5 name="quran" size={18} color="#15803d" />
              <Text style={styles.sectionTitle}>Original Audio</Text>
            </View>

            <View style={styles.optionsRow}>
              {!isOriginalUpload && (
                <TouchableOpacity
                  onPress={async () => {
                    const didPick = await pickAudio(setOriginalAudio);
                    setIsOriginalUpload(didPick);
                  }}
                  style={styles.uploadButton}
                >
                  <MaterialIcons name="upload-file" size={24} color="#16a34a" />
                  <Text style={styles.buttonText}>Upload</Text>
                </TouchableOpacity>
              )}

              {!originalAudio && !isOriginalUpload && (
                <Animatable.View
                  animation={isOriginalRecording ? "pulse" : undefined}
                  iterationCount="infinite"
                  style={styles.flexOne}
                >
                  <TouchableOpacity
                    onPress={() => recordAudio(true)}
                    style={
                      isOriginalRecording
                        ? styles.recordingButton
                        : styles.recordButton
                    }
                  >
                    <Ionicons
                      name={isOriginalRecording ? "stop" : "mic"}
                      size={24}
                      color={isOriginalRecording ? "#ef4444" : "#16a34a"}
                    />
                    <Text
                      style={
                        isOriginalRecording
                          ? styles.recordingText
                          : styles.buttonText
                      }
                    >
                      {isOriginalRecording ? "Recording..." : "Record"}
                    </Text>
                  </TouchableOpacity>
                </Animatable.View>
              )}
            </View>

            {originalAudio && (
              <AudioBar
                audio={originalAudio}
                onPlay={() => playSound(originalAudio)}
                onPause={pauseSound}
                isPlaying={isPlaying && currentlyPlaying === originalAudio}
                setAudio={setIsOriginalUpload}
                setIslUpload={setOriginalAudio}
              />
            )}
          </View>

          <View style={styles.divider} />

          <View style={styles.sectionContainer}>
            <View style={styles.sectionHeaderRow}>
              <FontAwesome5 name="microphone-alt" size={18} color="#15803d" />
              <Text style={styles.sectionTitle}>Your Recitation</Text>
            </View>

            <View style={styles.optionsRow}>
              {!isUserUpload && (
                <TouchableOpacity
                  onPress={async () => {
                    const didPick = await pickAudio(setUserAudio);
                    setIsUserUpload(didPick);
                  }}
                  style={styles.uploadButton}
                >
                  <MaterialIcons name="upload-file" size={24} color="#16a34a" />
                  <Text style={styles.buttonText}>Upload</Text>
                </TouchableOpacity>
              )}

              {!userAudio && !isUserUpload && (
                <Animatable.View
                  animation={isUserRecording ? "pulse" : undefined}
                  iterationCount="infinite"
                  style={styles.flexOne}
                >
                  <TouchableOpacity
                    onPress={() => recordAudio(false)}
                    style={
                      isUserRecording
                        ? styles.recordingButton
                        : styles.recordButton
                    }
                  >
                    <Ionicons
                      name={isUserRecording ? "stop" : "mic"}
                      size={24}
                      color={isUserRecording ? "#ef4444" : "#16a34a"}
                    />
                    <Text
                      style={
                        isUserRecording
                          ? styles.recordingText
                          : styles.buttonText
                      }
                    >
                      {isUserRecording ? "Recording..." : "Record"}
                    </Text>
                  </TouchableOpacity>
                </Animatable.View>
              )}
            </View>

            {userAudio && (
              <AudioBar
                audio={userAudio}
                onPlay={() => playSound(userAudio)}
                onPause={pauseSound}
                isPlaying={isPlaying && currentlyPlaying === userAudio}
                setAudio={setUserAudio}
                setIslUpload={setIsUserUpload}
              />
            )}
          </View>

          <TouchableOpacity
            onPress={similaritySearch}
            disabled={
              compareDTWMutation.isLoading || !originalAudio || !userAudio
            }
            style={[
              styles.compareButton,
              (compareDTWMutation.isLoading || !originalAudio || !userAudio) &&
                styles.disabledButton,
            ]}
          >
            <LinearGradient
              colors={["#15803d", "#166534"]}
              style={styles.compareButtonGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              {compareDTWMutation.isLoading ? (
                <View style={styles.loadingContainer}>
                  <ActivityIndicator color="#ffffff" size={20} />
                  <Text style={styles.compareButtonText}>Processing...</Text>
                </View>
              ) : (
                <View style={styles.buttonContentRow}>
                  <MaterialIcons name="compare" size={20} color="#ffffff" />
                  <Text style={styles.compareButtonText}>
                    Compare Recitations
                  </Text>
                </View>
              )}
            </LinearGradient>
          </TouchableOpacity>

          <View style={styles.infoContainer}>
            <Text style={styles.infoTitle}>How it works</Text>
            <Text style={styles.infoText}>
              Our advanced algorithm compares your recitation with the original,
              analyzing rhythm, pronunciation, and timing to provide you with a
              similarity score.
            </Text>
          </View>
        </ScrollView>
      </LinearGradient>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: "#fff",
  },
  container: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
    backgroundColor: "#fff",
    borderColor: "#e5e7eb",
  },
  backButton: {
    padding: 8,
    borderRadius: 20,
    backgroundColor: "#f0fdf4",
  },
  headerTitle: {
    flex: 1,
    fontSize: 20,
    color: "#15803d",
    fontWeight: "600",
    textAlign: "center",
  },
  headerSpacer: {
    width: 40,
  },
  introContainer: {
    padding: 16,
    backgroundColor: "#f0fdf4",
    borderBottomWidth: 1,
    borderColor: "#dcfce7",
  },
  introText: {
    fontSize: 16,
    color: "#166534",
    lineHeight: 22,
  },
  scrollContent: {
    padding: 20,
  },
  sectionContainer: {
    marginBottom: 24,
    backgroundColor: "#ffffff",
    borderRadius: 12,
    padding: 16,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 3,
    elevation: 2,
  },
  sectionHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "bold",
    marginLeft: 8,
    color: "#15803d",
  },
  divider: {
    height: 1,
    backgroundColor: "#e5e7eb",
    marginVertical: 8,
  },
  optionsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 16,
  },
  uploadButton: {
    borderWidth: 2,
    borderColor: "#dcfce7",
    paddingVertical: 16,
    borderRadius: 8,
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 8,
    backgroundColor: "#f0fdf4",
  },
  recordButton: {
    borderWidth: 2,
    borderColor: "#dcfce7",
    paddingVertical: 16,
    borderRadius: 8,
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#f0fdf4",
  },
  recordingButton: {
    borderWidth: 2,
    borderColor: "#fecaca",
    paddingVertical: 16,
    borderRadius: 8,
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#fef2f2",
  },

  audioPreviewContainer: {
    marginTop: 12,
    padding: 8,
    borderWidth: 1,
    borderColor: "#d1d5db",
    borderRadius: 8,
    position: "relative",
    backgroundColor: "#f9fafb",
  },
  removeButton: {
    position: "absolute",
    top: 4,
    left: 4,
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: "#ef4444",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 2,
  },
  removeButtonText: {
    color: "#fff",
    fontSize: 16,
    lineHeight: 18,
    fontWeight: "bold",
  },
  audioInfo: {
    flexDirection: "row",
    alignItems: "center",
  },
  audioName: {
    flex: 1,
    marginHorizontal: 8,
    color: "#111827",
  },
  playButton: {
    backgroundColor: "#15803d",
    padding: 6,
    borderRadius: 4,
  },

  buttonText: {
    fontSize: 16,
    marginTop: 8,
    color: "#166534",
    fontWeight: "500",
  },
  recordingText: {
    fontSize: 16,
    marginTop: 8,
    color: "#ef4444",
    fontWeight: "500",
  },
  audioBarContainer: {
    borderRadius: 8,
    overflow: "hidden",
  },
  audioBarGradient: {
    flexDirection: "row",
    alignItems: "center",
    padding: 16,
    borderRadius: 8,
  },
  audioTitle: {
    flex: 1,
    marginRight: 8,
    fontSize: 16,
    fontWeight: "500",
    color: "#166534",
  },
  playButton: {
    backgroundColor: "#16a34a",
    borderRadius: 20,
    width: 40,
    height: 40,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  pauseButton: {
    backgroundColor: "#ef4444",
    borderRadius: 20,
    width: 40,
    height: 40,
    justifyContent: "center",
    alignItems: "center",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  compareButton: {
    borderRadius: 8,
    overflow: "hidden",
    marginTop: 8,
    marginBottom: 24,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 2,
  },
  compareButtonGradient: {
    paddingVertical: 16,
    alignItems: "center",
  },
  disabledButton: {
    opacity: 0.5,
  },
  flexOne: {
    flex: 1,
  },
  loadingContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
  },
  buttonContentRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  compareButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
  },
  infoContainer: {
    backgroundColor: "#f8fafc",
    padding: 16,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: "#15803d",
    marginBottom: 24,
  },
  infoTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#15803d",
    marginBottom: 8,
  },
  infoText: {
    fontSize: 14,
    color: "#4b5563",
    lineHeight: 20,
  },
});
