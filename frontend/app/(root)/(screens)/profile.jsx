import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  SafeAreaView,
  TouchableOpacity,
  StyleSheet,
  Image,
  ActivityIndicator,
  Alert,
  TextInput,
  Modal,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Switch
} from "react-native";
import { useRouter } from "expo-router";
import { Ionicons } from "@expo/vector-icons";
import { useAuth } from "../../../context/AuthContext";
import authService from "../../../utils/services/authService";
import * as ImagePicker from "expo-image-picker";
import * as SecureStore from "expo-secure-store";
import { useQueryClient } from '@tanstack/react-query'
import usePrayerSettingsStore from "../../../zustand/MonthlyPrayerStore";

export default function ProfileScreen() {
  const router = useRouter();
  // Update this line at the top of your component
  const { user, logout, refreshUserData } = useAuth();
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [editField, setEditField] = useState("");
  const [editValue, setEditValue] = useState("");
  const [updating, setUpdating] = useState(false);
  const [avatarUrl, setAvatarUrl] = useState(null);
  const [avatarLoading, setAvatarLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [previewVisible, setPreviewVisible] = useState(false);


  const queryClient = useQueryClient()
  const { schoolShift, setSchoolShift } = usePrayerSettingsStore()

  const onToggleShift = (value) => {
    setSchoolShift(value ? 'shafi' : 'hanfi');
    queryClient.invalidateQueries(['prayer-calendar']);
  };

  useEffect(() => {
    // If not authenticated, redirect to login
    if (!user) {
      router.replace("/(auth)/sign-in");
      return;
    }

    // Fetch user data
    fetchUserData();
  }, [user]);

  useEffect(() => {
    // Load avatar if available
    if (userData?.avatar) {
      loadAvatar(userData.avatar);
    }
  }, [userData]);

  // Update the loadAvatar function
  const loadAvatar = async (avatarId) => {
    try {
      const url = await authService.getAvatarUrl(avatarId);
      if (url) {
        setAvatarUrl(url);
      }
    } catch (error) {}
  };

  // Update the avatar image component
  {
    avatarLoading ? (
      <View style={styles.avatarLoadingContainer}>
        <ActivityIndicator size="large" color="#0b8c5c" />
        <Text style={styles.avatarLoadingText}>Updating...</Text>
      </View>
    ) : avatarUrl ? (
      <Image
        source={{
          uri: avatarUrl,
          headers: {
            Authorization: `Bearer ${SecureStore.getItemAsync("access_token")}`,
          },
          cache: "reload",
        }}
        style={styles.avatar}
        onError={(e) =>
          console.log("Image loading error:", e.nativeEvent.error)
        }
      />
    ) : (
      <View style={styles.avatarPlaceholder}>
        <Text style={styles.avatarText}>
          {userData?.name ? userData.name.charAt(0).toUpperCase() : "U"}
        </Text>
      </View>
    );
  }
  const fetchUserData = async () => {
    try {
      setLoading(true);
      // Try to get user data from API
      const data = await authService.getUserData();

      if (data) {
        setUserData(data);
      } else {
        // Fallback to local data if API fails
        setUserData({
          name: user.name || "User",
          email: user.email || "No email available",
          avatar: user.avatar || null,
        });
      }
    } catch (error) {
      // Fallback to local data
      setUserData({
        name: user.name || "User",
        email: user.email || "No email available",
        avatar: user.avatar || null,
      });
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      router.replace("/(root)/(screens)/home");
    } catch (error) {
      Alert.alert("Error", "Failed to logout. Please try again.");
    }
  };

  const openEditModal = (field, value) => {
    setEditField(field);
    setEditValue(value || "");
    setEditModalVisible(true);
  };

  // Update the handleUpdateProfile function to use refreshUserData
  const handleUpdateProfile = async () => {
    if (!editValue.trim()) {
      Alert.alert("Error", `Please enter a valid ${editField}`);
      return;
    }

    try {
      setUpdating(true);

      // Create update data object
      const updateData = {};
      updateData[editField] = editValue;

      // Call API to update user
      const result = await authService.updateUser(updateData);

      if (result) {
        // Update local state
        setUserData((prev) => ({
          ...prev,
          [editField]: editValue,
        }));

        // Refresh user data in the auth context
        if (refreshUserData) {
          await refreshUserData();
        }

        Alert.alert("Success", "Profile updated successfully");
        setEditModalVisible(false);

        // If email was updated, inform the user they might need to log in again
        if (editField === "email") {
          Alert.alert(
            "Email Updated",
            "Your email has been updated. You may need to log out and log back in for all features to work correctly.",
            [{ text: "OK" }]
          );
        }
      } else {
        Alert.alert("Error", "Failed to update profile");
      }
    } catch (error) {
      let errorMessage = "Failed to update profile";
      if (error.response && error.response.data && error.response.data.detail) {
        errorMessage = error.response.data.detail;
      }
      Alert.alert("Error", errorMessage);

      // If we get a 401 error, the token might be invalid
      if (error.response && error.response.status === 401) {
        Alert.alert(
          "Session Expired",
          "Your session has expired. Please log in again.",
          [
            {
              text: "OK",
              onPress: async () => {
                await logout();
                router.replace("/(auth)/sign-in");
              },
            },
          ]
        );
      }
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <SafeAreaView style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#0b8c5c" />
        <Text style={styles.loadingText}>Loading profile...</Text>
      </SafeAreaView>
    );
  }

  const handlePickAvatar = async () => {
    try {
      // Request permission
      const { status } =
        await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (status !== "granted") {
        Alert.alert(
          "Permission Required",
          "Please grant permission to access your photos"
        );
        return;
      }

      // Launch image picker
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.7,
      });

      if (!result.canceled && result.assets && result.assets.length > 0) {
        const selectedImage = result.assets[0];

        // Show preview first
        setPreviewImage(selectedImage.uri);
        setPreviewVisible(true);
      }
    } catch (error) {
      console.log("Error picking image:", error);
      Alert.alert("Error", "Failed to pick image");
    }
  };

  const handleAvatarUpdate = async () => {
    if (!previewImage) return;

    try {
      setAvatarLoading(true);
      setPreviewVisible(false);

      // Update user with new avatar
      const updateResult = await authService.updateUser({
        avatar: previewImage,
      });

      if (updateResult) {
        // Update avatar URL
        if (updateResult.avatar) {
          // Force refresh the avatar by adding a timestamp query parameter
          const timestamp = new Date().getTime();
          const url = await authService.getAvatarUrl(
            updateResult.avatar,
            timestamp
          );
          setAvatarUrl(url);
        }

        // Refresh user data in the auth context
        if (refreshUserData) {
          await refreshUserData();
        }

        Alert.alert("Success", "Profile picture updated successfully");
      } else {
        Alert.alert("Error", "Failed to update profile picture");
      }
    } catch (error) {
      let errorMessage = "Failed to update profile picture";
      if (error.response && error.response.data && error.response.data.detail) {
        errorMessage = error.response.data.detail;
      }
      Alert.alert("Error", errorMessage);
    } finally {
      setAvatarLoading(false);
      setPreviewImage(null);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.profileContainer}>
        <TouchableOpacity
          style={styles.avatarContainer}
          onPress={handlePickAvatar}
          disabled={avatarLoading}
        >
          {avatarLoading ? (
            <View style={styles.avatarLoadingContainer}>
              <ActivityIndicator size="large" color="#0b8c5c" />
              <Text style={styles.avatarLoadingText}>Updating...</Text>
            </View>
          ) : avatarUrl ? (
            <Image source={{ uri: avatarUrl }} style={styles.avatar} />
          ) : (
            <View style={styles.avatarPlaceholder}>
              <Text style={styles.avatarText}>
                {userData?.name ? userData.name.charAt(0).toUpperCase() : "U"}
              </Text>
            </View>
          )}
          <View style={styles.editAvatarBadge}>
            <Ionicons name="camera" size={16} color="white" />
          </View>
        </TouchableOpacity>

        <Text style={styles.userName}>{userData?.name || "User"}</Text>
        <Text style={styles.userEmail}>
          {userData?.email || "No email available"}
        </Text>

        <View style={styles.infoContainer}>
          <TouchableOpacity
            style={styles.infoItem}
            onPress={() => openEditModal("name", userData?.name)}
          >
            <Ionicons name="person-outline" size={24} color="#0b8c5c" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoLabel}>Name</Text>
              <Text style={styles.infoValue}>
                {userData?.name || "Not available"}
              </Text>
            </View>
            <Ionicons name="pencil-outline" size={20} color="#6b7280" />
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.infoItem}
            onPress={() => openEditModal("email", userData?.email)}
          >
            <Ionicons name="mail-outline" size={24} color="#0b8c5c" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoLabel}>Email</Text>
              <Text style={styles.infoValue}>
                {userData?.email || "Not available"}
              </Text>
            </View>
            <Ionicons name="pencil-outline" size={20} color="#6b7280" />
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.infoItem}
            onPress={() => openEditModal("password", "")}
          >
            <Ionicons name="lock-closed-outline" size={24} color="#0b8c5c" />
            <View style={styles.infoTextContainer}>
              <Text style={styles.infoLabel}>Password</Text>
              <Text style={styles.infoValue}>••••••••</Text>
            </View>
            <Ionicons name="pencil-outline" size={20} color="#6b7280" />
          </TouchableOpacity>

          {/* ─── School Shift Setting ─── */}
          <View style={styles.settingItem}>
            <Text style={styles.settingLabel}>School of Thought</Text>
            <View style={styles.switchRow}>
              <Text>Hanafi</Text>
              <Switch
                value={schoolShift === "shafi"}
                onValueChange={onToggleShift}
                thumbColor={schoolShift==="shafi" ?"#0b8c5c" : "#464646"}
                trackColor={{
                  false: "#d1d5db", 
                  true:  "#9ae6b4" 
                }}
              />
              <Text>Shafi</Text>
            </View>
          </View>
        </View>

        <TouchableOpacity style={styles.logoutButton} onPress={handleLogout}>
          <Ionicons name="log-out-outline" size={20} color="white" />
          <Text style={styles.logoutText}>Logout</Text>
        </TouchableOpacity>
      </View>

      {/* Edit Profile Modal */}
      <Modal
        visible={editModalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setEditModalVisible(false)}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === "ios" ? "padding" : "height"}
          style={styles.modalContainer}
        >
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                Edit {editField.charAt(0).toUpperCase() + editField.slice(1)}
              </Text>
              <TouchableOpacity onPress={() => setEditModalVisible(false)}>
                <Ionicons name="close" size={24} color="#6b7280" />
              </TouchableOpacity>
            </View>

            <TextInput
              style={styles.modalInput}
              value={editValue}
              onChangeText={setEditValue}
              placeholder={`Enter your ${editField}`}
              secureTextEntry={editField === "password"}
              autoCapitalize={editField === "name" ? "words" : "none"}
              keyboardType={editField === "email" ? "email-address" : "default"}
            />

            <TouchableOpacity
              style={styles.updateButton}
              onPress={handleUpdateProfile}
              disabled={updating}
            >
              {updating ? (
                <ActivityIndicator color="white" size="small" />
              ) : (
                <Text style={styles.updateButtonText}>Update</Text>
              )}
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </Modal>

      <Modal
        visible={previewVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setPreviewVisible(false)}
      >
        <View style={styles.previewModalContainer}>
          <View style={styles.previewModalContent}>
            <View style={styles.previewModalHeader}>
              <Text style={styles.previewModalTitle}>
                Preview Profile Picture
              </Text>
              <TouchableOpacity onPress={() => setPreviewVisible(false)}>
                <Ionicons name="close" size={24} color="#6b7280" />
              </TouchableOpacity>
            </View>

            <View style={styles.previewImageContainer}>
              {previewImage && (
                <Image
                  source={{ uri: previewImage }}
                  style={styles.previewImage}
                  resizeMode="cover"
                />
              )}
            </View>

            <View style={styles.previewButtonsContainer}>
              <TouchableOpacity
                style={[styles.previewButton, styles.cancelButton]}
                onPress={() => {
                  setPreviewVisible(false);
                  setPreviewImage(null);
                }}
              >
                <Text style={styles.cancelButtonText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.previewButton, styles.confirmButton]}
                onPress={handleAvatarUpdate}
              >
                <Text style={styles.confirmButtonText}>Use This Photo</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "white",
  },

  loadingContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "white",
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: "#0b8c5c",
  },
  avatarLoadingContainer: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "rgba(11, 140, 92, 0.1)",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#0b8c5c",
  },
  avatarLoadingText: {
    marginTop: 8,
    fontSize: 12,
    color: "#0b8c5c",
    fontWeight: "600",
  },
  previewModalContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.7)",
    padding: 20,
  },
  previewModalContent: {
    backgroundColor: "white",
    borderRadius: 16,
    width: "100%",
    maxWidth: 350,
    padding: 20,
  },
  previewModalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
  },
  previewModalTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#1f2937",
  },
  previewImageContainer: {
    alignItems: "center",
    marginBottom: 20,
  },
  previewImage: {
    width: 200,
    height: 200,
    borderRadius: 100,
    borderWidth: 3,
    borderColor: "#0b8c5c",
  },
  previewButtonsContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  previewButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  cancelButton: {
    backgroundColor: "#f3f4f6",
    marginRight: 8,
  },
  confirmButton: {
    backgroundColor: "#0b8c5c",
    marginLeft: 8,
  },
  cancelButtonText: {
    color: "#4b5563",
    fontSize: 16,
    fontWeight: "600",
  },
  confirmButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },

  profileContainer: {
    flex: 1,
    alignItems: "center",
    paddingTop: 30,
    paddingHorizontal: 20,
  },
  avatarContainer: {
    marginBottom: 20,
    position: "relative",
  },
  avatar: {
    width: 120,
    height: 120,
    borderRadius: 60,
    borderWidth: 3,
    borderColor: "#0b8c5c",
  },
  avatarPlaceholder: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: "#e6f7ef",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#0b8c5c",
  },
  avatarText: {
    fontSize: 48,
    fontWeight: "bold",
    color: "#0b8c5c",
  },
  editAvatarBadge: {
    position: "absolute",
    bottom: 0,
    right: 0,
    backgroundColor: "#0b8c5c",
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "white",
  },
  userName: {
    fontSize: 24,
    fontWeight: "bold",
    color: "#1f2937",
    marginBottom: 5,
  },
  userEmail: {
    fontSize: 16,
    color: "#6b7280",
    marginBottom: 30,
  },
  infoContainer: {
    width: "100%",
    marginBottom: 40,
  },
  infoItem: {
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
  },
  infoTextContainer: {
    flex: 1,
    marginLeft: 15,
  },
  infoLabel: {
    fontSize: 14,
    color: "#6b7280",
  },
  infoValue: {
    fontSize: 16,
    color: "#1f2937",
    fontWeight: "500",
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderTopWidth: 1,
    borderColor: '#e5e7eb',
  },
  settingLabel: {
    fontSize: 16,
    color: '#374151',
  },
  switchRow: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  logoutButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "#0b8c5c",
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
    marginTop: "auto",
    marginBottom: 30,
    width: "100%",
  },
  logoutText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
    marginLeft: 8,
  },
  modalContainer: {
    flex: 1,
    justifyContent: "flex-end",
    backgroundColor: "rgba(0,0,0,0.5)",
  },
  modalContent: {
    backgroundColor: "white",
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    paddingBottom: Platform.OS === "ios" ? 40 : 20,
  },
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 20,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: "#1f2937",
  },
  modalInput: {
    borderWidth: 1,
    borderColor: "#e5e7eb",
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    marginBottom: 20,
    backgroundColor: "#f9fafb",
  },
  updateButton: {
    backgroundColor: "#0b8c5c",
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: "center",
  },
  updateButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
});
