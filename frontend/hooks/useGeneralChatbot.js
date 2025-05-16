import { useMutation } from '@tanstack/react-query';
import { getGeneralChatbotAnswer } from '../utils/services/apis';
import {useChat} from "../context/ChatContext"

export const useGeneralChatbotMutation = () => {
  const {language} = useChat();
  return useMutation({
    mutationFn: (question) => getGeneralChatbotAnswer(question,language),
    onError: (error) => {
      console.error('Genral Chatbot Mutation Error:', error);
      return error.message;
    }
  });
};