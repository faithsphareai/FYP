import { View, Text, Image, TouchableOpacity } from "react-native";
import { useRouter } from "expo-router";
import { images , icons} from "../../constants";

export default function WelcomeScreen() {
  const router = useRouter();

  return (
    <View className="flex-1 bg-white justify-between items-center p-6">
    {/* Spacer */}
    <View className="flex-1" />

    <View className="flex-row items-center gap-x-2 mb-8">
      <Image
          source={images.faithspherelogo}
          resizeMode="contain"
          style={{width: 50, height: 50}}
      />
      <Text className="text-xl font-bold">FaithSphere</Text>
    </View>

    {/* Welcome Image */}
    <Image
      source={images.welcome}
      resizeMode="contain"
      style={{ width: 400, height: 400 }}
    />

    {/* Welcome Text */}
    <View className="justify-center items-center px-4 my-8">
      <Text className="text-2xl font-bold text-center mb-2">Welcome to FaithSphere</Text>
      <Text className="text-base text-gray-500 text-center">
        Your gateway to a world of knowledge and enlightenment
      </Text>
    </View>

    {/* Spacer */}
    <View className="flex-1" />

    {/* Get Started Button */}
    <TouchableOpacity
      onPress={() => router.push("(auth)/sign-up")}
      className="w-full bg-[#0b8c5c] py-2 px-6 rounded-xl items-center flex-row justify-center gap-x-3 shadow-md"
    >
      <Text className="text-white text-xl font-bold">Get Started</Text>
      <Image source={icons.arrowRight} className="w-5 h-5 tint-white" />
    </TouchableOpacity>
  </View>
);
}