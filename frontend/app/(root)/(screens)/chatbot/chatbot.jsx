import React, { useRef, useEffect, useState } from "react";
import { StyleSheet } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { ChatHeader } from "../../../../components/ChatHeader";
import { ChatBubble } from "../../../../components/ChatBubble";
import { ChatInput } from "../../../../components/ChatInput";
import { ChatProvider, useChat } from "../../../../context/ChatContext";
import { FlatList } from "react-native";
import PropTypes from "prop-types";
import { useChatActions } from "../../../../hooks/useChatActions";
import LanguageSelectorModal from "../../../../components/LanguageSelector";

const ChatbotScreenContent = ({ chatContext, title }) => {
  const { messages, clearMessages } = useChat();
  const { sendMessage, startNewQuiz } = useChatActions();
  const flatListRef = useRef(null);
  const [languageModalVisible, setLanguageModalVisible] = useState(true); // Show on load

  // Are we awaiting an answer to a question?
  // const lastBot = [...messages].reverse().find((m) => m.sender === "bot");
  // const isQuestionPending = !!lastBot?.isQuestion && chatContext === "quiz";

  useEffect(() => {
    if (flatListRef.current && messages.length > 0) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  }, [messages]);

  const handleNewTopic = async () => {
    if (chatContext === "quiz") {
      await clearMessages();
      startNewQuiz();
    }
  };

  const renderMessage = ({ item }) => (
    <ChatBubble message={item} isUser={item.sender === "user"} />
  );

  const keyExtractor = (item) => {
    return `${item.timestamp}-${item.sender}-${
      item.id || Math.random().toString()
    }`;
  };

  return (
    <SafeAreaView style={styles.container}>
      {(chatContext === "quran" || chatContext === "hadith") && (
        <LanguageSelectorModal
          visible={languageModalVisible}
          onClose={() => setLanguageModalVisible(false)}
        />
      )}
      <ChatHeader
        context={title}
        onNewTopic={handleNewTopic}
        onResetChat={clearMessages}
      />
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderMessage}
        keyExtractor={keyExtractor}
        contentContainerStyle={styles.chatContainer}
        inverted={false}
        onLayout={() => flatListRef.current?.scrollToEnd({ animated: false })}
      />
      {/* <ChatInput
        onSend={sendMessage}
        buttonLabel={
          chatContext === "quiz"
            ? isQuestionPending
              ? "Grade"
              : "Send"
            : "Send"
        }
      /> */}
      <ChatInput onSend={sendMessage} />
    </SafeAreaView>
  );
};

ChatbotScreenContent.propTypes = {
  chatContext: PropTypes.string,
  title: PropTypes.string,
};

const ChatBot = ({ chatContext, title }) => (
  <ChatProvider chatContext={chatContext}>
    <ChatbotScreenContent chatContext={chatContext} title={title} />
  </ChatProvider>
);

ChatBot.propTypes = {
  chatContext: PropTypes.string,
  title: PropTypes.string,
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
  },
  chatContainer: {
    padding: 16,
    flexGrow: 1,
  },
  timestamp: {
    fontSize: 10,
    marginTop: 4,
    opacity: 0.7,
  },
  userTimestamp: {
    color: "rgba(255, 255, 255, 0.7)",
  },
  botTimestamp: {
    color: "rgba(0, 0, 0, 0.7)",
  },
});

export default ChatBot;
