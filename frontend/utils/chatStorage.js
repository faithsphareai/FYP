import AsyncStorage from "@react-native-async-storage/async-storage";

export const loadChatHistory = async (chatContext) => {
  try {
    const key = `chat_history_${chatContext}`;
    const data = await AsyncStorage.getItem(key);
    return data ? JSON.parse(data) : [];
  } catch (error) {
    console.error("Error loading chat history:", error);
    return [];
  }
};
export const saveChatHistory = async (chatContext, messages) => {
  try {
    const key = `chat_history_${chatContext}`;
    await AsyncStorage.setItem(key, JSON.stringify(messages));
  } catch (error) {
    console.error("Error saving chat history:", error);
  }
};
