import React, { useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  ScrollView,
  ActivityIndicator,
  Alert,
  TextInput,
  StyleSheet,
  Platform,
  Clipboard,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useRouter } from "expo-router";
import * as ImagePicker from "expo-image-picker";
import * as DocumentPicker from "expo-document-picker";
import {
  Ionicons,
  MaterialIcons,
  MaterialCommunityIcons,
} from "@expo/vector-icons";
import * as Animatable from "react-native-animatable";
import { useOcrMutation } from "../../../hooks/useOcrMutation";
import { Camera } from 'expo-camera';


export default function OCRScreen() {
  const router = useRouter();
  const [file, setFile] = useState(null);
  const [fileType, setFileType] = useState(null); // 'image' or 'pdf'
  const [recognizedText, setRecognizedText] = useState("");

  // Using the OCR mutation hook
  const ocrMutation = useOcrMutation();
  const isProcessing = ocrMutation.isLoading;

  const handleBack = () => {
    router.back();
  };

  async function requestCameraPermission() {
    const { status } = await Camera.requestCameraPermissionsAsync();
    return status === "granted";
  }

  async function requestLibraryPermission() {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    return status === "granted";
  }
  // const pickImage = async () => {
  //   try {
  //     const result = await ImagePicker.launchImageLibraryAsync({
  //       mediaTypes: ImagePicker.MediaTypeOptions.Images,
  //       allowsEditing: true,
  //       quality: 1,
  //     });

  //     if (!result.canceled) {
  //       setFile(result.assets[0].uri);
  //       setFileType('image');
  //       setRecognizedText('');
  //     }
  //   } catch (error) {
  //     console.error('Error picking image:', error);
  //     Alert.alert('Error', 'Failed to pick image');
  //   }
  // };
  const pickImage = async () => {
    const options = ["Take Photo", "Choose from Library", "Cancel"];
    const cancelButtonIndex = 2;

    const handleSelection = async (index) => {
      let result;
      try {
        if (index === 0) {
          // CAMERA
          const hasCamera = await requestCameraPermission();
          if (!hasCamera) {
            Alert.alert(
              "Permission required",
              "Camera access is needed to take photos"
            );
            return;
          }
          result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            quality: 1,
          });
        } else if (index === 1) {
          // LIBRARY
          const hasLibrary = await requestLibraryPermission();
          if (!hasLibrary) {
            Alert.alert(
              "Permission required",
              "Storage access is needed to select photos"
            );
            return;
          }
          result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            quality: 1,
          });
        } else {
          return; // cancelled
        }

        if (!result.canceled) {
          setFile(result.assets[0].uri);
          setFileType("image");
          setRecognizedText("");
        }
      } catch (error) {
        console.error("Error picking image:", error);
        Alert.alert("Error", "Failed to pick image");
      }
    };

    if (Platform.OS === "ios") {
      const { ActionSheetIOS } = await import("react-native");
      ActionSheetIOS.showActionSheetWithOptions(
        {
          options,
          cancelButtonIndex,
        },
        handleSelection
      );
    } else {
      Alert.alert(
        "Select Image Source",
        undefined,
        [
          { text: options[0], onPress: () => handleSelection(0) },
          { text: options[1], onPress: () => handleSelection(1) },
          { text: options[2], style: "cancel" },
        ],
        { cancelable: true }
      );
    }
  };

  const pickPdf = async () => {
    try {
      const result = await DocumentPicker.getDocumentAsync({
        type: "application/pdf",
        copyToCacheDirectory: true,
      });

      if (result.canceled === false) {
        setFile(result.assets[0].uri);
        setFileType("pdf");
        setRecognizedText("");
      }
    } catch (error) {
      console.error("Error picking PDF:", error);
      Alert.alert("Error", "Failed to pick PDF file");
    }
  };

  const processFile = async () => {
    if (!file) {
      Alert.alert("No file", "Please select an image or PDF first");
      return;
    }

    try {
      // Call the OCR API using our mutation
      const extractedText = await ocrMutation.mutateAsync({
        fileUri: file,
        fileType: fileType,
      });

      setRecognizedText(extractedText);
    } catch (error) {
      Alert.alert(
        "OCR Error",
        error.message || "Failed to process file. Please try again."
      );
    }
  };

  const copyToClipboard = () => {
    Clipboard.setString(recognizedText);
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBack}>
          <Ionicons name="arrow-back" size={24} color="#15803d" />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>OCR Scanner</Text>
        <View style={styles.headerRightPlaceholder} />
      </View>

      <ScrollView style={styles.content}>
        <Text style={styles.sectionTitle}>
          Upload an image or PDF to extract text
        </Text>

        {/* File Preview */}
        {file && fileType === "image" ? (
          <Animatable.View
            animation="fadeIn"
            duration={500}
            style={styles.previewContainer}
          >
            <Image
              source={{ uri: file }}
              style={styles.imagePreview}
              resizeMode="contain"
            />
          </Animatable.View>
        ) : file && fileType === "pdf" ? (
          <Animatable.View
            animation="fadeIn"
            duration={500}
            style={styles.pdfPreviewContainer}
          >
            <MaterialCommunityIcons
              name="file-pdf-box"
              size={64}
              color="#ef4444"
            />
            <Text style={styles.pdfPreviewText}>PDF Document Selected</Text>
          </Animatable.View>
        ) : (
          <View style={styles.emptyPreviewContainer}>
            <MaterialIcons name="upload-file" size={64} color="#9ca3af" />
            <Text style={styles.emptyPreviewText}>No file selected</Text>
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionButtonsContainer}>
          {/* <TouchableOpacity onPress={pickImage} style={styles.actionButton}>
            <MaterialIcons name="photo-library" size={24} color="#15803d" />
            <Text style={styles.actionButtonText}>Image</Text>
          </TouchableOpacity> */}
          <TouchableOpacity onPress={pickImage} style={styles.actionButton}>
            <MaterialIcons name="photo-library" size={24} color="#15803d" />
            <Text style={styles.actionButtonText}>Image</Text>
          </TouchableOpacity>

          <TouchableOpacity
            onPress={pickPdf}
            style={[styles.actionButton, { marginLeft: 8 }]}
          >
            <MaterialCommunityIcons
              name="file-pdf-box"
              size={24}
              color="#15803d"
            />
            <Text style={styles.actionButtonText}>PDF</Text>
          </TouchableOpacity>
        </View>

        {/* Process Button */}
        <TouchableOpacity
          onPress={processFile}
          disabled={!file || isProcessing}
          style={[
            styles.processButton,
            !file || isProcessing
              ? styles.disabledButton
              : styles.enabledButton,
          ]}
        >
          {isProcessing ? (
            <View style={styles.processingContainer}>
              <ActivityIndicator size="small" color="#ffffff" />
              <Text style={styles.processButtonText}>Processing...</Text>
            </View>
          ) : (
            <Text style={styles.processButtonText}>Extract Text</Text>
          )}
        </TouchableOpacity>

        {/* Recognized Text (Disabled TextArea) */}
        <View style={styles.textResultContainer}>
          <View style={styles.textResultHeader}>
            <Text style={styles.textResultTitle}>Extracted Text:</Text>
            <TouchableOpacity
              onPress={copyToClipboard}
              disabled={!recognizedText}
              style={[
                styles.copyButton,
                !recognizedText && styles.disabledCopyButton,
              ]}
            >
              <MaterialIcons
                name="content-copy"
                size={20}
                color={recognizedText ? "#15803d" : "#9ca3af"}
              />
              <Text
                style={[
                  styles.copyButtonText,
                  !recognizedText && styles.disabledCopyButtonText,
                ]}
              >
                Copy
              </Text>
            </TouchableOpacity>
          </View>
          <ScrollView
            style={styles.textResultScrollView}
            nestedScrollEnabled={true}
          >
            <Text style={styles.textResultInput}>
              {recognizedText || "Extracted text will appear here"}
            </Text>
          </ScrollView>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff", // green-50
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 16,
    paddingVertical: 16,
    borderBottomWidth: 1,
    backgroundColor: "#fff", // green-50
    borderBottomColor: "#e5e7eb", // gray-200
  },
  headerTitle: {
    flex: 1,
    fontSize: 18,
    color: "#15803d", // green-800
    fontWeight: "600",
    textAlign: "center",
  },
  headerRightPlaceholder: {
    width: 24,
  },
  content: {
    flex: 1,
    backgroundColor: "white",
    padding: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "500",
    color: "#1f2937", // gray-800
    marginBottom: 16,
  },
  previewContainer: {
    marginBottom: 16,
  },
  imagePreview: {
    width: "100%",
    height: 256, // h-64
    borderRadius: 8,
  },
  pdfPreviewContainer: {
    marginBottom: 16,
    width: "100%",
    height: 256, // h-64
    backgroundColor: "#f3f4f6", // gray-100
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
  },
  pdfPreviewText: {
    color: "#374151", // gray-700
    marginTop: 8,
    fontWeight: "500",
  },
  emptyPreviewContainer: {
    width: "100%",
    height: 256, // h-64
    backgroundColor: "#f3f4f6", // gray-100
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 16,
  },
  emptyPreviewText: {
    color: "#6b7280", // gray-500
    marginTop: 8,
  },
  actionButtonsContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 24,
  },
  actionButton: {
    backgroundColor: "#f0fdf4", // green-50
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    flex: 1,
    alignItems: "center",
    flexDirection: "row",
    justifyContent: "center",
  },
  actionButtonText: {
    color: "#15803d", // green-800
    fontWeight: "500",
    marginLeft: 8,
  },
  processButton: {
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
    marginBottom: 24,
  },
  disabledButton: {
    backgroundColor: "#d1d5db", // gray-300
  },
  enabledButton: {
    backgroundColor: "#0b8c5c", // custom green
  },
  processingContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  processButtonText: {
    color: "white",
    fontWeight: "600",
    marginLeft: 8,
  },
  textResultContainer: {
    marginBottom: 16,
    flex: 1,
  },
  textResultHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  copyButton: {
    flexDirection: "row",
    alignItems: "center",
    padding: 8,
    borderRadius: 6,
    backgroundColor: "#f0fdf4",
  },
  disabledCopyButton: {
    backgroundColor: "#f3f4f6",
  },
  copyButtonText: {
    marginLeft: 4,
    color: "#15803d",
    fontWeight: "500",
  },
  disabledCopyButtonText: {
    color: "#9ca3af",
  },
  textResultScrollView: {
    maxHeight: 160,
    backgroundColor: "#f3f4f6",
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "#e5e7eb",
  },
  textResultInput: {
    padding: 12,
    color: "#374151",
    fontFamily: Platform.OS === "ios" ? "Menlo" : "monospace",
  },
  textResultInput: {
    padding: 12,
    color: "#374151",
    textAlignVertical: "top",
  },
});
