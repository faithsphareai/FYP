import React from 'react';
import { useLocalSearchParams } from 'expo-router';
import ChatBot from './chatbot/chatbot';

export default function ChatbotRoute() {
  const params = useLocalSearchParams();
  // Ensure params are strings
  const chatContext = typeof params.chatContext === 'string' ? params.chatContext : '';
  const title = typeof params.title === 'string' ? params.title : 'Chat';
  
  return <ChatBot chatContext={chatContext} title={title} />;
}