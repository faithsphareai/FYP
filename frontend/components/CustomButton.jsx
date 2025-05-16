
import { TouchableOpacity, Text, Image } from "react-native";
import { icons } from "../constants";

export function Button({ title, onPress }) {
    return (
        <TouchableOpacity 
        onPress={onPress} 
        className="bg-[#0b8c5c] py-4 px-6 rounded-lg flex-row justify-between items-center mb-4"
      >
        <Text className="text-white text-lg font-semibold">{title}</Text>
        <Image 
          source={icons.arrowRight}  
          className="w-6 h-6 tint-white"
        />
      </TouchableOpacity>
    );
  }