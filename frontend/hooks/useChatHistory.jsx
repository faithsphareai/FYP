import { useState, useEffect } from 'react';
import { loadChatHistory } from '../utils/chatStorage';

export const useChatHistory = (chatContext) => {
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const loadHistory = async () => {
      const history = await loadChatHistory(chatContext);
      setMessages(history);
    };
    loadHistory();
  }, [chatContext]);

  return { messages, setMessages };
};